from typing import Callable, Generic, TypeVar, Any
import numpy as np
from threading import Event, Lock, Condition
from queue import Queue
from models import BaseModel, MODELS

MAX_INDEX_QUEUE_SIZE = 50
MAX_INDEX_BATCH_SIZE = 20

T = TypeVar('T')
class CLIPQueueProcessor(Generic[T]):
    class Queue:
        def __init__(self, model_index):
            self.model_index = model_index
            self.q: list[tuple[Callable[[np.ndarray], None], T]] = []
            self.full_event = Event()
            self.is_queued = False
            self.is_queued_lock = Condition()

    def __init__(self):
        self.queues: dict[int, CLIPQueueProcessor.Queue] = {}
        self.queues_lock = Lock()
        self.qq: Queue[CLIPQueueProcessor.Queue] = Queue()

    def add(self, model_index: int, callback: Callable[[np.ndarray], None], data: T):
        if model_index < 0 or model_index >= len(MODELS):
            raise IndexError("Model index out of range.")

        with self.queues_lock:
            if model_index not in self.queues:
                self.queues[model_index] = CLIPQueueProcessor.Queue(model_index)
            q = self.queues[model_index]

        with q.is_queued_lock:
            while len(q.q) >= MAX_INDEX_QUEUE_SIZE:
                q.is_queued_lock.wait()
            q.q.append((callback, data))
            if not q.is_queued:
                self.qq.put(q)
                q.is_queued = True

    def process(self, model: BaseModel, data: list[T]) -> list[np.ndarray]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def worker(self):
        while True:
            q = self.qq.get()
            with q.is_queued_lock:
                items, remaining = q.q[:MAX_INDEX_BATCH_SIZE], q.q[MAX_INDEX_BATCH_SIZE:]
                q.q = remaining
                if remaining:
                    self.qq.put(q)
                else:
                    q.is_queued = False
                q.is_queued_lock.notify_all()

            callbacks = []
            inputs = []
            for callback, _input in items:
                callbacks.append(callback)
                inputs.append(_input)

            for i, vector in enumerate(self.process(MODELS[q.model_index].model, inputs)):
                callbacks[i](vector)

class VCLIPQueueProcessor(CLIPQueueProcessor[np.ndarray]):
    def process(self, model: BaseModel, data: list[np.ndarray]) -> list[np.ndarray]:
        return model.process_videos(data)

class TCLIPQueueProcessor(CLIPQueueProcessor[str]):
    def process(self, model: BaseModel, data: list[str]) -> list[np.ndarray]:
        return model.process_texts(data)