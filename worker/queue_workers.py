from typing import Callable, Generic, TypeVar, Any
import numpy as np
from threading import Event, Lock, Condition
from queue import Queue, Empty
from models import BaseModel, MODELS

MAX_INDEX_QUEUE_SIZE = 50 # maximum length for each model's processing queue
MAX_INDEX_BATCH_SIZE = 20 # maximum number of items processed by the model at a time

# uses generic to avoid code duplication while retaining type hinting
# video worker takes ndarrays as input (raw frames) while text worker takes strings
T = TypeVar('T')
class CLIPQueueProcessor(Generic[T]):
    class Queue: # per-model queue
        def __init__(self, model_index):
            self.model_index = model_index
            self.q: list[tuple[Callable[[np.ndarray], None], T]] = []
            self.full_event = Event()
            self.is_queued = False
            self.is_queued_lock = Condition()

    def __init__(self, max_queue_size: int = None):
        self.max_queue_size: int = max_queue_size
        self.queues: dict[int, CLIPQueueProcessor.Queue] = {}
        self.queues_lock = Lock()
        self.qq: Queue[CLIPQueueProcessor.Queue] = Queue() # queue of queues!

    def add(self, model_index: int, data: T, callback: Callable[[np.ndarray], None]):
        if model_index < 0 or model_index >= len(MODELS):
            raise IndexError("Model index out of range.")

        with self.queues_lock:
            # if no queue exists for this model, make one
            if model_index not in self.queues:
                self.queues[model_index] = CLIPQueueProcessor.Queue(model_index)
            q = self.queues[model_index]

        with q.is_queued_lock:
            if self.max_queue_size is not None:
                while len(q.q) >= self.max_queue_size:
                    q.is_queued_lock.wait() # this releases the lock until awoken by notify(_all)
            q.q.append((callback, data)) # add our item to the queue
            if not q.is_queued: # if this per-model queue was removed from the main queue (happens when empty), add it
                self.qq.put(q)
                q.is_queued = True

    def clear(self):
        while not self.qq.empty():
            try:
                q = self.qq.get(block=False)
            except Empty:
                return
            with q.is_queued_lock:
                q.q.clear()
                q.is_queued = False

    def process(self, model: BaseModel, data: list[T]) -> list[np.ndarray]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def worker(self):
        while True:
            q = self.qq.get() # fetch a per-model queue from the main queue
            with q.is_queued_lock:
                # fetch up to batch size of items
                items, remaining = q.q[:MAX_INDEX_BATCH_SIZE], q.q[MAX_INDEX_BATCH_SIZE:]
                q.q = remaining # update the queue to only the items we didn't take
                if remaining: # if more items remain, put it back in the main queue
                    self.qq.put(q)
                else:
                    q.is_queued = False # mark it as no longer queued so it'll be put back on when items are added
                q.is_queued_lock.notify_all() # if anything was waiting for space to be available, wake it up

            # process batch
            callbacks = []
            inputs = []
            for callback, _input in items:
                callbacks.append(callback)
                inputs.append(_input)

            for i, vector in enumerate(self.process(MODELS[q.model_index].model, inputs)):
                callbacks[i](vector) # run each callback

class VCLIPQueueProcessor(CLIPQueueProcessor[np.ndarray]):
    def __init__(self):
        super().__init__(MAX_INDEX_QUEUE_SIZE)

    def process(self, model: BaseModel, data: list[np.ndarray]) -> list[np.ndarray]:
        return model.process_videos(data)

class TCLIPQueueProcessor(CLIPQueueProcessor[str]):
    def process(self, model: BaseModel, data: list[str]) -> list[np.ndarray]:
        return model.process_texts(data)
