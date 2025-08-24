import faiss
import os
import struct
import numpy as np
from readerwriterlock import rwlock
from enum import IntEnum, auto
from models import MODELS
from threading import Lock, Event
from typing import Any

# faiss index wrapper with automatic saving/loading and named vectors
class Index:
    def __init__(self, dim: int, name: str):
        # if saved index exists, load it, else make new
        if os.path.exists(f"indexes/{name}.index") and os.path.exists(f"indexes/{name}.names"):
            index: faiss.IndexFlatL2 = faiss.read_index(f"indexes/{name}.index")

            if not isinstance(index, faiss.IndexFlatL2):
                raise ValueError(f"Index {name} is not of type IndexFlatL2.")
            
            if index.d != dim:
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

class State(IntEnum):
    DOWNLOADING = auto()
    PROCESSING = auto()
    FAILED = auto()

# combines multiple faiss indexes (one for each model) into one container for a single user
# also tracks the state of each indexing job
class UserIndexStore:
    def __init__(self, user_key: str):
        self.user_key = user_key
        self.pending_states_lock = Lock()
        self.pending_states: dict[tuple[str, int], tuple[State, str]] = {}
        self.indexes: list[Index] = []
        # an index per model
        for i, model in enumerate(MODELS):
            self.indexes.append(Index(model.dim, f"{user_key}_{i}"))

    def set_state(self, name: str, model_index: int, state: State, data: Any = None):
        self.pending_states[(name, model_index)] = (state, data)

    def del_state(self, name: str, model_index: int):
        self.pending_states.pop((name, model_index), None)

    def get_state(self, name: str, model_index: int) -> tuple[State, Any]:
        return self.pending_states.get((name, model_index), (None, None))

    # stores a CLIP vector
    def add(self, model_index: int, name: str, vector: np.ndarray):
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        with self.pending_states_lock:
            self.indexes[model_index].add([name], np.array([vector]))
            self.pending_states.pop((name, model_index), None)

    # fetches a CLIP vector
    def get(self, model_index: int, name: str) -> np.ndarray:
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        return self.indexes[model_index].get(name)

    # finds the k closest vectors to a given vector for a given model index
    def search(self, model_index: int, vectors: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        return self.indexes[model_index].search(vectors, k)

    # given the name of a GIF, returns each model index as either failed, missing, downloading, processing, or completed
    def status(self, name: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
        remaining = set(range(len(MODELS)))

        processing = set()
        downloading = set()
        missing = set()
        failed = set()

        completed = set(filter(lambda m: self.indexes[m].is_present(name), remaining))
        remaining -= completed
        if not remaining:
            return failed, missing, downloading, processing, completed

        for model_idx in remaining:
            status, _ = self.get_state(name, model_idx)
            if status == State.PROCESSING:
                processing.add(model_idx)
            elif status == State.DOWNLOADING:
                downloading.add(model_idx)
            elif status == State.FAILED:
                failed.add(model_idx)
            else:
                missing.add(model_idx)
        return failed, missing, downloading, processing, completed

    # removes a name from the faiss index for a given model index
    def remove(self, names: set[str], model_index: int):
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        self.indexes[model_index].remove(names)
        for name in names:
            self.pending_states.pop((name, model_index), None)

    def remove_not_present(self, names: set[str]):
        for model_index in range(len(MODELS)):
            with self.pending_states_lock:
                # remove any indexed names that are not in the given names
                names_indexed_this = set(self.indexes[model_index].names)
                self.remove(names_indexed_this - names, model_index)

                # remove any pending names that are not in the given names
                for name in set([name for name, model_idx in self.pending_states if model_idx == model_index]) - names:
                    state, data = self.get_state(name, model_index)
                    if state in (State.DOWNLOADING, State.PROCESSING):
                        target = IndexDestination(name, self) # construct target search item
                        # might be able to do this search with set operations as they should have same hash and eq
                        found: IndexDestination = None
                        for dest in data:
                            if dest == target:
                                found = dest
                                break
                        else:
                            continue
                        with found.lock:
                            if found.event.is_set(): # we were too late, state moved from processing -> completed, now remove from index
                                self.remove({name}, model_index)
                                continue
                            else:
                                # still processing, set event so destination will be skipped
                                found.event.set()
                        data.discard(found)

# container class for target user indexes after processing
class IndexDestination:
    def __init__(self, name: str, user_index: UserIndexStore):
        self.name = name
        self.user_index = user_index
        self.lock = Lock()
        self.event = Event()

    def __hash__(self):
        return hash((self.name, id(self.user_index)))

    def __eq__(self, other):
        return self.name == other.name and self.user_index is other.user_index
