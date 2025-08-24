import faiss
import os
import struct
import numpy as np
from readerwriterlock import rwlock
from enum import IntEnum, auto
from models import MODELS
from threading import Lock, Event
from typing import Any

class Index:
    def __init__(self, dim: int, name: str):
        if os.path.exists(f"indexes/{name}.index") and os.path.exists(f"indexes/{name}.names"):
            index: faiss.IndexFlatL2 = faiss.read_index(f"indexes/{name}.index")

            if not isinstance(index, faiss.IndexFlatL2):
                raise ValueError(f"Index {name} is not of type IndexFlatL2.")
            
            if index.d != dim:
                raise ValueError(f"Index dimension {index.d} does not match provided dimension {dim}.")
            
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
        self.lock = rwlock.RWLockWrite()

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

        with self.lock.gen_wlock():
            self.index.add(vectors)
            self.names.extend(names)
        self.save()

    def remove(self, names: set[str]):
        with self.lock.gen_wlock():
            idxes = [self.names.index(name) for name in names if name in self.names]
            if not idxes: return

            self.index.remove_ids(np.array(idxes))
            for idx in sorted(idxes, reverse=True):
                self.names.pop(idx)

        self.save()

    def search(self, vectors: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError("Vectors must be a 2D array with shape (n, dim).")
        
        with self.lock.gen_rlock():
            distances, indices = self.index.search(vectors, k)
            results = []
            for i in range(len(vectors)):
                result = [(self.names[idx], distances[i][j].item()) for j, idx in enumerate(indices[i]) if idx >= 0]
                results.append(result)
        return results
    
    def get(self, name: str) -> np.ndarray:
        with self.lock.gen_rlock():
            if name not in self.names:
                return None
        
            idx = self.names.index(name)
            return self.index.reconstruct(idx)

    def is_present(self, name: str) -> bool:
        with self.lock.gen_rlock():
            return name in self.names
    
    def save(self):
        with self.lock.gen_rlock():
            faiss.write_index(self.index, f"indexes/{self.name}.index.writing")
            with open(f"indexes/{self.name}.names.writing", 'wb') as f:
                for name in self.names:
                    f.write(struct.pack('H', len(name)))
                    f.write(name.encode('utf-8'))
            os.replace(f"indexes/{self.name}.index.writing", f"indexes/{self.name}.index")
            os.replace(f"indexes/{self.name}.names.writing", f"indexes/{self.name}.names")

    def clear(self):
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

class UserIndexStore:
    def __init__(self, user_key: str):
        self.pending_states: dict[tuple[str, int], tuple[State, str]] = {}
        self.indexes: list[Index] = []
        self.user_key = user_key
        for i, model in enumerate(MODELS):
            self.indexes.append(Index(model.dim, f"{user_key}_{i}"))

    def set_state(self, name: str, model_index: int, state: State, data: Any = None):
        self.pending_states[(name, model_index)] = (state, data)

    def del_state(self, name: str, model_index: int):
        self.pending_states.pop((name, model_index), None)

    def get_state(self, name: str, model_index: int) -> tuple[State, Any]:
        return self.pending_states.get((name, model_index), (None, None))

    def add(self, model_index: int, name: str, vector: np.ndarray):
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        self.indexes[model_index].add([name], np.array([vector]))
        self.pending_states.pop((name, model_index), None)

    def get(self, model_index: int, name: str) -> np.ndarray:
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        return self.indexes[model_index].get(name)

    def search(self, model_index: int, vectors: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        return self.indexes[model_index].search(vectors, k)

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

    def remove(self, names: set[str], model_index: int):
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        self.indexes[model_index].remove(names)
        for name in names:
            self.pending_states.pop((name, model_index), None)

    def remove_not_present(self, names: set[str]):
        for model_index in range(len(MODELS)):
            names_indexed_this = set(self.indexes[model_index].names)
            self.remove(names_indexed_this - names, model_index)
            for name in set([name for name, model_idx in self.pending_states if model_idx == model_index]) - names:
                state, data = self.get_state(name, model_index)
                if state in (State.DOWNLOADING, State.PROCESSING):
                    target = IndexDestination(name, self)
                    found: IndexDestination = None
                    for dest in data:
                        if dest == target:
                            found = dest
                            break
                    else:
                        continue
                    with found.lock:
                        if found.event.is_set():
                            self.remove({name}, model_index)
                            continue
                        else:
                            found.event.set()
                    data.discard(found)

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
