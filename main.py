from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import os
import numpy as np
import struct
import requests
from threading import Thread, Lock, Event, Condition
from io import BytesIO
from PIL import Image
import cv2
from collections import defaultdict
from enum import IntEnum, auto
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Generic, TypeVar, Any
from queue import Queue
from readerwriterlock import rwlock

from models.base import BaseModel
from models.videoclip_xl_v2 import VideoCLIP_XL_v2
from models.xclip import XClip

MAX_INDEX_QUEUE_SIZE = 50
MAX_INDEX_BATCH_SIZE = 20

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
            raise ValueError("Vectors must be a 2D array with shape (n, dim).")

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

class ResizeMode(IntEnum):
    STRETCH = auto()
    CENTER_CROP = auto()

class Model:
    def __init__(self, name: str, model: BaseModel, dim: int, res: tuple[int, int], fcount: int, resize_mode: ResizeMode):
        self.name = name
        self.model = model
        self.dim = dim
        self.res = res
        self.fcount = fcount
        self.resize_mode = resize_mode

MODELS = [
    Model("VideoCLIP-XL-v2", VideoCLIP_XL_v2(), 768, (224, 224), 8, ResizeMode.STRETCH),
    Model("X-CLIP", XClip(), 512, (224, 224), 8, ResizeMode.CENTER_CROP),
]

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

    def process(self, model: BaseModel, data: T) -> list[np.ndarray]:
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

class State(IntEnum):
    DOWNLOADING = auto()
    PROCESSING = auto()
    FAILED = auto()

class UserIndexStore:
    def __init__(self, user_key: str):
        self.pending_states: dict[tuple[str, int], tuple[State, str]] = {}
        self.indexes: list[Index] = []
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

user_indexes = {}

def get_user_index(user_key: str) -> UserIndexStore:
    if user_key not in user_indexes:
        user_indexes[user_key] = UserIndexStore(user_key)
    return user_indexes[user_key]

def is_valid_user_key(user_key: str) -> bool:
    if len(user_key) != 32:
        return False
    for char in user_key:
        if char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_":
            return False
    return True

class Indexer:

    class VJob:
        def __init__(self, model_index: int, media_src: str, destinations: set[IndexDestination]):
            self.model_index = model_index
            self.media_src = media_src
            self.destinations = destinations
            self.lock = Lock()

    class TJob:
        def __init__(self, model_index: int, text: str):
            self.model_index = model_index
            self.text = text
            self.lock = Lock()
            self.event = Event()
            self.output = None

    def __init__(self):
        self.in_progress_v_lock = Lock()
        self.in_progress_v: dict[tuple[int, str], Indexer.VJob] = {}
        self.in_progress_v_dl: dict[str, defaultdict[int, set[IndexDestination]]] = {}

        self.in_progress_t_lock = Lock()
        self.in_progress_t: dict[tuple[int, str], Indexer.TJob] = {}

        self.global_v_index = UserIndexStore("global_v")
        self.global_t_index = UserIndexStore("global_t")

        self.failed_media_dl = set()

        self.download_executor = ThreadPoolExecutor(max_workers=5)

        self.vclip_queue_processor = VCLIPQueueProcessor()
        self.tclip_queue_processor = TCLIPQueueProcessor()
        Thread(target=self.vclip_queue_processor.worker, daemon=True).start()
        Thread(target=self.tclip_queue_processor.worker, daemon=True).start()

    def tclip_complete_callback(self, job: TJob, vector: np.ndarray):
        with self.in_progress_t_lock:
            del self.in_progress_t[(job.model_index, job.text)]
            self.global_t_index.add(job.model_index, job.text, vector)
        job.output = vector
        job.event.set()

    def get_text_vectors(self, models: list[int], text: str):
        vectors: (list[Indexer.TJob] | list[np.ndarray]) = [None] * len(models)
        has_vector_idxes = set()
        with self.in_progress_t_lock:
            for i, model_idx in enumerate(models):
                ret = self.in_progress_t.get((model_idx, text))
                if ret is not None:
                    vectors[i] = ret
                    continue

                vec = self.global_t_index.get(model_idx, text)
                if vec is not None:
                    vectors[i] = vec
                    has_vector_idxes.add(i)
                    continue

                job = Indexer.TJob(model_idx, text)
                self.in_progress_t[(model_idx, text)] = job
                self.tclip_queue_processor.add(model_idx, lambda vec, job=job: self.tclip_complete_callback(job, vec), text)
                vectors[i] = job
        
        if len(has_vector_idxes) != len(models):
            for i in has_vector_idxes:
                job = Indexer.TJob(models[i], text)
                job.output = vectors[i]
                job.event.set()
                vectors[i] = job

        return vectors

    def vclip_complete_callback(self, job: VJob, model_index: int, vector: np.ndarray):
        with self.in_progress_v_lock:
            del self.in_progress_v[(model_index, job.media_src)]
            self.global_v_index.add(model_index, job.media_src, vector)
            for destination in job.destinations:
                with destination.lock:
                    if destination.event.is_set():
                        continue
                    destination.user_index.add(model_index, destination.name, vector)
                    destination.event.set()

    def _download_media(self, url: str, media_src: str):
        failed = True
        required_models = self.in_progress_v_dl[media_src]
        popped = False
        try:
            r = requests.get(url, stream=True)
            if "content-length" not in r.headers or "content-type" not in r.headers:
                return

            if int(r.headers["content-length"]) > 50 * 1024 * 1024:
                return

            content_type = r.headers["content-type"]
            if content_type not in ("image/gif", "image/png", "video/mp4"):
                return
            
            data = BytesIO(r.content)

            del r

            with self.in_progress_v_lock:
                del self.in_progress_v_dl[media_src]
                popped = True

            processed_media = {}
            for model_idx, destinations in required_models.items():
                model = MODELS[model_idx]
                req = (model.fcount, model.res, model.resize_mode)
                if req not in processed_media:
                    processed = self.resize_media(data, content_type, *req)
                    data.seek(0)
                    if not processed:
                        return
                    processed_media[req] = processed
                job = Indexer.VJob(model_idx, media_src, destinations)
                with self.in_progress_v_lock:
                    self.in_progress_v[(model_idx, media_src)] = job
                self.vclip_queue_processor.add(model_idx, lambda vector, job=job, model_idx=model_idx: self.vclip_complete_callback(job, model_idx, vector), processed_media[req])
            failed = False
        finally:
            with self.in_progress_v_lock:
                if failed: self.failed_media_dl.add(url)
                for model_idx, destinations in required_models.items():
                    for destination in destinations:
                        destination.user_index.set_state(destination.name, model_idx, State.FAILED if failed else State.PROCESSING, destinations)
                if not popped: del self.in_progress_v_dl[media_src]

    def enqueue(self, user_index: UserIndexStore, media_src: str, url: str, name: str, models: set[int]):
        with self.in_progress_v_lock:
            models = set(filter(lambda m: user_index.get(m, name) is None, models))
            if not models: return

            def from_global_filter(model_index: int) -> bool:
                res = self.global_v_index.get(model_index, media_src)
                if res is not None:
                    user_index.add(model_index, name, res)
                    return False
                return True
            models = set(filter(from_global_filter, models))
            if not models: return

            destination = IndexDestination(name, user_index)

            def subscribe_to_embedding_filter(model_index: int) -> bool:
                vjob = self.in_progress_v.get((model_index, media_src))
                if vjob is None:
                    return True
                vjob.destinations.add(destination)
                user_index.set_state(name, model_index, State.PROCESSING, vjob.destinations)
                return False
            models = set(filter(subscribe_to_embedding_filter, models))
            if not models: return

            if url in self.failed_media_dl:
                for model_index in models:
                    user_index.set_state(name, model_index, State.FAILED)
                return
            
            downloading = self.in_progress_v_dl.get(media_src)
            if downloading is not None:
                for model_index in models:
                    downloading[model_index].add(destination)
                    user_index.set_state(name, model_index, State.DOWNLOADING, downloading[model_index])
                return
            
            self.in_progress_v_dl[media_src] = defaultdict(set)
            for model_index in models:
                self.in_progress_v_dl[media_src][model_index].add(destination)
                user_index.set_state(name, model_index, State.DOWNLOADING, self.in_progress_v_dl[media_src][model_index])
        #Thread(target=self._download_media, args=(media_src,), daemon=True).start()
        self.download_executor.submit(self._download_media, url, media_src)

    def resize_image(self, image: Image.Image, resolution: tuple[int, int], resize_mode: ResizeMode) -> Image.Image:
        if resize_mode == ResizeMode.STRETCH:
            return image.resize(resolution)
        elif resize_mode == ResizeMode.CENTER_CROP:
            new_width, new_height = image.size
            new_top = 0
            new_left = 0
            if new_width < new_height:
                new_top = (new_height - new_width) // 2
                new_height = new_width
            else:
                new_left = (new_width - new_height) // 2
                new_width = new_height
            return image.resize(resolution, box=(new_left, new_top, new_left + new_width, new_top + new_height))
        else:
            raise ValueError("Invalid resize mode")

    def resize_mat_cv2(self, mat: np.ndarray, resolution: tuple[int, int], resize_mode: ResizeMode) -> np.ndarray:
        if resize_mode == ResizeMode.STRETCH:
            return cv2.resize(mat, resolution)
        elif resize_mode == ResizeMode.CENTER_CROP:
            new_height, new_width = mat.shape[:2]
            new_top = 0
            new_left = 0
            if new_width < new_height:
                new_top = (new_height - new_width) // 2
                new_height = new_width
            else:
                new_left = (new_width - new_height) // 2
                new_width = new_height
            return cv2.resize(mat[new_top:new_top + new_height, new_left:new_left + new_width], resolution)
        else:
            raise ValueError("Invalid resize mode")

    def resize_media(self, data: BytesIO, content_type: str, nframes: int, resolution: tuple[int, int], resize_mode: ResizeMode) -> list[np.ndarray]:
        frames = []
        if content_type in ("image/gif", "image/png"):
            image = Image.open(data)
            if content_type == "image/gif":
                if image.n_frames == 0:
                    return []
                if image.n_frames < nframes:
                    for i in range(image.n_frames):
                        image.seek(i)
                        frames.append(np.array(self.resize_image(image.convert("RGB"), resolution, resize_mode)))
                    frames.extend([frames[-1]] * (nframes - len(frames)))
                else:
                    step = image.n_frames / nframes
                    for i in range(nframes):
                        image.seek(int(i * step))
                        frames.append(np.array(self.resize_image(image.convert("RGB"), resolution, resize_mode)))
            else:
                frames = [np.array(self.resize_image(image.convert("RGB"), resolution, resize_mode))] * nframes
        else:
            video = cv2.VideoCapture(source=data, apiPreference=cv2.CAP_FFMPEG, params=[])
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                return []
            if total_frames < nframes:
                for _ in range(total_frames):
                    ret, frame = video.read()
                    if not ret:
                        break
                    frames.append(self.resize_mat_cv2(frame, resolution, resize_mode))
                frames.extend([frames[-1]] * (nframes - len(frames)))
            else:
                step = total_frames / nframes
                for i in range(nframes):
                    video.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
                    ret, frame = video.read()
                    if not ret:
                        break
                    frames.append(self.resize_mat_cv2(frame, resolution, resize_mode))
        return frames

indexer = Indexer()

app = Flask(__name__)
CORS(app)

def error(message: str):
    return jsonify({"status": "error", "message": message})

@app.route('/<user_key>/index', methods=['POST'])
def index(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    payload = request.get_json()
    if "media_srcs" not in payload or "names" not in payload or "models" not in payload:
        return error("Missing required fields"), 400

    models_list: list[int] = payload["models"]
    if not isinstance(models_list, list):
        return error("models must be a list"), 400

    if len(models_list) == 0 or any(not isinstance(m, int) or m < 0 or m >= len(MODELS) for m in models_list):
        return error("Invalid model indexes"), 400

    models: set[int] = set(models_list)

    names: list[str] = payload["names"]
    if not isinstance(names, list) or not names:
        return error("names must be a non-empty list of strings"), 400
    for name in names:
        if not isinstance(name, str) or not name.strip():
            return error("Each name must be a non-empty string"), 400
        if len(name) > 512:
            return error("Each name must not exceed 512 characters"), 400

    media_srcs: list[str] = payload["media_srcs"]
    if not isinstance(media_srcs, list) or len(media_srcs) != len(names):
        return error("media_srcs must be a list of strings with the same length as names"), 400
    
    user_index = get_user_index(user_key)
    for i, item in enumerate(media_srcs):
        if not isinstance(item, str) or not item.strip():
            return error("Each media_src must be a non-empty string"), 400
        if len(item) > 2000:
            return error("media_src exceeds 2000 characters"), 400

        if item.startswith("http://"): domain_offset = 7
        elif item.startswith("https://"): domain_offset = 8
        else: return error(f"Invalid media_src: {item}"), 400

        path_idx = item.find('/', domain_offset)
        if path_idx == -1:
            path_idx = len(item)

        domain = item[domain_offset:path_idx]

        if not domain or len(domain) > 256 or (not domain.endswith(".discordapp.net") and not domain == "media.tenor.co"):
            return error(f"Invalid domain in media_src: {item}"), 400

    for i, url in enumerate(media_srcs):
        params_idx = url.find("?")
        if params_idx != -1:
            media_src = url[:params_idx]
        else:
            media_src = url
        indexer.enqueue(user_index, media_src, url, names[i], models)

    user_index.remove_not_present(set(names))

    return jsonify({"status": "ok", "message": "Indexing started"}), 202

@app.route('/<user_key>/status', methods=['GET'])
def status(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    name = request.args.get('name')
    if not name:
        return error("Missing 'name' parameter"), 400

    failed, missing, downloading, processing, completed = get_user_index(user_key).status(name)
    return jsonify({
        "status": "ok",
        "failed": list(failed),
        "missing": list(missing),
        "downloading": list(downloading),
        "processing": list(processing),
        "completed": list(completed)
    }), 200
    
@app.route('/<user_key>/search', methods=['GET'])
def search(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    text = request.args.get('text')
    if not text:
        return error("Missing 'text' parameter"), 400
    
    res = indexer.get_text_vectors(list(range(len(MODELS))), text)
    vectors = []
    if isinstance(res[0], Indexer.TJob):
        for i in range(len(MODELS)):
            res[i].event.wait()
            vectors.append(res[i].output)
    else:
        vectors = res

    return jsonify({
        "status": "ok",
        "results": {i: get_user_index(user_key).search(i, np.array([vectors[i]]), k=10)[0] for i in range(len(MODELS))}
        }), 200
