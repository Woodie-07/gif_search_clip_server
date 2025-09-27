import faiss
import os
import struct
import numpy as np
from readerwriterlock import rwlock
from enum import IntEnum, auto
from threading import Lock, Event
from typing import Any
from collections import defaultdict

# faiss index wrapper with automatic saving/loading and named vectors
class Index:
    def __init__(self, dim: int, name: str):
        # if saved index exists, load it, else make new
        if os.path.exists(f"indexes/{name}.index") and os.path.exists(f"indexes/{name}.names"):
            index: faiss.IndexFlatL2 = faiss.read_index(f"indexes/{name}.index")

            if not isinstance(index, faiss.IndexFlatL2):
                raise ValueError(f"Index {name} is not of type IndexFlatL2.")
            
            if dim is not None and index.d != dim:
                raise ValueError(f"Index dimension {index.d} does not match provided dimension {dim}.")

            # if the process was killed after overwriting index but before overwriting names, overwrite names to maintain consistency
            if os.path.exists(f"indexes/{name}.names.writing") and not os.path.exists(f"indexes/{name}.index.writing"):
                os.replace(f"indexes/{name}.names.writing", f"indexes/{name}.names")
            
            # names file is a list of strings each prefixed by 2 bytes for their length
            names = []
            with open(f"indexes/{name}.names", 'rb') as f:
                while True:
                    length_bytes = f.read(2)
                    if not length_bytes:
                        break
                    length = struct.unpack('H', length_bytes)[0]
                    names.append(f.read(length).decode('utf-8'))

            self.index = index
            self.names: list[str] = names
        else:
            if dim is None:
                raise ValueError("dim unspecified for new index")
            self.index = faiss.IndexFlatL2(dim)
            self.names: list[str] = []

        self.name = name
        self.lock = rwlock.RWLockWrite() # lock that allows multiple readers but only one writer

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def add(self, names: list[str], vectors: np.ndarray):
        if len(vectors) != len(names):
            raise ValueError("The number of names must match the number of vectors.")
        
        for name in names:
            if name in self.names:
                raise ValueError(f"Name '{name}' already exists in the index.")
            
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError(f"Vectors must be a 2D array with shape (n, {self.index.d}), not {vectors.shape}.")

        with self.lock.gen_wlock(): # get an exclusive lock while we add to the index
            self.index.add(vectors)
            self.names.extend(names)
        self.save()

    def remove(self, names: set[str]):
        with self.lock.gen_wlock():
            # find the indexes of the given names, if present, then remove from the index and names list
            idxes = [self.names.index(name) for name in names if name in self.names]
            if not idxes: return

            self.index.remove_ids(np.array(idxes))
            for idx in sorted(idxes, reverse=True):
                self.names.pop(idx)

        self.save()

    def search(self, vectors: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError("Vectors must be a 2D array with shape (n, dim).")
        
        # for each of the given vectors, find the k closest names along with their distances
        with self.lock.gen_rlock():
            distances, indices = self.index.search(vectors, k)
            results = []
            for i in range(len(vectors)):
                result = [(self.names[idx], distances[i][j].item()) for j, idx in enumerate(indices[i]) if idx >= 0]
                results.append(result)
        return results
    
    def get(self, name: str) -> np.ndarray:
        # get the vector associated with the given name
        with self.lock.gen_rlock():
            if name not in self.names:
                return None
        
            idx = self.names.index(name)
            return self.index.reconstruct(idx)

    def is_present(self, name: str) -> bool:
        # check if the given name is in the index
        with self.lock.gen_rlock():
            return name in self.names
    
    def save(self):
        with self.lock.gen_rlock():
            # write the index and names to a file ending with .writing so we don't overwrite the old copy yet
            faiss.write_index(self.index, f"indexes/{self.name}.index.writing")
            with open(f"indexes/{self.name}.names.writing", 'wb') as f:
                for name in self.names:
                    f.write(struct.pack('H', len(name)))
                    f.write(name.encode('utf-8'))
            
            # now that the files have been written entirely, overwrite the old copies
            os.replace(f"indexes/{self.name}.index.writing", f"indexes/{self.name}.index")
            os.replace(f"indexes/{self.name}.names.writing", f"indexes/{self.name}.names")

    def clear(self):
        # clear the index and associated names
        with self.lock.gen_wlock():
            self.index.reset()
            self.names.clear()

        if os.path.exists(f"indexes/{self.name}.index"):
            os.remove(f"indexes/{self.name}.index")

        if os.path.exists(f"indexes/{self.name}.names"):
            os.remove(f"indexes/{self.name}.names")

class JobStatus(IntEnum):
    DOWNLOADING = 0
    FAILED = 1
    PROCESSING = 2
    COMPLETED = 3

# combines multiple faiss indexes (one for each model) into one container for a single user
# also tracks the state of each indexing job
class UserIndexStore:
    def __init__(self, user_key: str):
        self.user_key = user_key
        self.pending_states_lock = Lock()
        self.pending_states: dict[str, dict[str, StateContainer]] = {}
        self.indexes: dict[str, Index] = {}

    def add_pending(self, name: str, model_name: str, state_container: "StateContainer"):
        with self.pending_states_lock:
            if name not in self.pending_states:
                self.pending_states[name] = {}
            self.pending_states[name][model_name] = state_container

    def remove_pending(self, name: str, model_name: str):
        with self.pending_states_lock:
            if name not in self.pending_states:
                return
            self.pending_states[name].pop(model_name)
            if not self.pending_states[name]:
                self.pending_states.pop(name)

    def get_state(self, name: str, model_name: str) -> JobStatus:
        try:
            return self.pending_states[name][model_name].get_state()
        except KeyError:
            return None

    # stores a CLIP vector
    def add(self, name: str, model_name: str, vector: np.ndarray):
        if model_name not in self.indexes:
            self.indexes[model_name] = Index(vector.shape[0], f"{self.user_key}_{model_name}")

        self.indexes[model_name].add([name], np.array([vector]))

        self.remove_pending(name, model_name)

    def _try_load_index(self, model_name: str) -> bool:
        if model_name not in self.indexes:
            try:
                self.indexes[model_name] = Index(None, f"{self.user_key}_{model_name}")
            except ValueError:
                return False
        return True

    # fetches a CLIP vector
    def get(self, model_name: str, name: str) -> np.ndarray:
        if not self._try_load_index(model_name):
            return None
        return self.indexes[model_name].get(name)

    def is_present(self, model_name: str, name: str) -> bool:
        if not self._try_load_index(model_name):
            return False
        return self.indexes[model_name].is_present(name)

    # finds the k closest vectors to a given vector for a given model
    def search(self, model_name: str, vectors: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if not self._try_load_index(model_name):
            return [[]] * len(vectors)

        return self.indexes[model_name].search(vectors, k)

    # given the name of a GIF, returns each model name as either failed, downloading, processing, or completed
    def status(self, name: str) -> tuple[set[str], set[str], set[str], set[str]]:
        remaining = set(self.indexes.keys())

        completed = set()
        processing = set()
        downloading = set()
        failed = set()

        completed = set(filter(lambda m: self.indexes[m].is_present(name), remaining))
        remaining -= completed
        if not remaining:
            return failed, downloading, processing, completed

        for model_name in remaining:
            status = self.get_state(name, model_name)
            {
                JobStatus.PROCESSING: processing,
                JobStatus.DOWNLOADING: downloading,
                JobStatus.FAILED: failed
            }[status].add(model_name)
        return failed, downloading, processing, completed
    
    def status_all(self) -> dict[str, tuple[set[str], set[str], set[str], set[str]]]:
        statuses: defaultdict[str, tuple[set[str], set[str], set[str], set[str]]] = defaultdict(lambda: (set(), set(), set(), set()))
        for model_name, index in self.indexes.items():
            for name in index.names:
                statuses[name][4].add(model_name)

        with self.pending_states_lock:
            for name, model_states in self.pending_states.items():
                for model_name, state_container in model_states.items():
                    status = state_container.get_state()
                    {
                        JobStatus.PROCESSING: statuses[name][2],
                        JobStatus.DOWNLOADING: statuses[name][1],
                        JobStatus.FAILED: statuses[name][0]
                    }[status].add(model_name)

        return statuses
    
    def status_counts(self) -> dict[str, tuple[int, int, int, int]]:
        counts: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for model_name, index in self.indexes.items():
            counts[model_name][3] += len(index.names)

        with self.pending_states_lock:
            print(self.pending_states)
            for model_states in self.pending_states.values():
                for model_name, state_container in model_states.items():
                    status = state_container.get_state()
                    if status == JobStatus.PROCESSING:
                        counts[model_name][2] += 1
                    elif status == JobStatus.DOWNLOADING:
                        counts[model_name][1] += 1
                    elif status == JobStatus.FAILED:
                        counts[model_name][0] += 1

        return {model: tuple(count) for model, count in counts.items()}
    
    # removes names from the faiss index for a given model name
    def remove(self, names: set[str], model: str):
        if model not in self.indexes:
            raise KeyError("Model not found.")

        self.indexes[model].remove(names)

    def remove_not_present(self, names: set[str]):
        # remove any indexed names that are not in the given names
        for model in self.indexes.keys():
            names_indexed_this = set(self.indexes[model].names)
            self.remove(names_indexed_this - names, model)

        # remove any pending names that are not in the given names
        for name in set(self.pending_states.keys()) - names:
            for state_container in self.pending_states[name].values():
                state_container.get_destinations().remove(IndexDestination(name, self))

# container class for target user indexes after processing
class IndexDestination:
    def __init__(self, name: str, user_index: UserIndexStore):
        self.name = name
        self.user_index = user_index

    def __hash__(self):
        return hash((self.name, id(self.user_index)))

    def __eq__(self, other):
        if not isinstance(other, IndexDestination): return False
        return self.name == other.name and self.user_index is other.user_index
    
    def index(self, model: str, vector: np.ndarray):
        self.user_index.add(self.name, model, vector)

class StateContainer:
    def __init__(self, data = None):
        self.state = JobStatus.DOWNLOADING
        self.dests: set[IndexDestination] = set()
        self.data = data

    def set_state(self, state: JobStatus):
        self.state = state

    def get_state(self) -> JobStatus:
        return self.state
    
    def index(self, model: str, vector: np.ndarray):
        for dest in self.dests:
            dest.index(model, vector)

    def add_destination(self, dest: IndexDestination):
        self.dests.add(dest)

    def get_destinations(self) -> set[IndexDestination]:
        return self.dests

    def get_data(self):
        return self.data
