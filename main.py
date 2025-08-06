from flask import Flask, request, jsonify
import faiss
import os
import numpy as np
import struct
import requests
from threading import Lock, Thread, Event
from io import BytesIO
from PIL import Image
import cv2
from collections import defaultdict
from models.base import BaseModel
from models.videoclip_xl_v2 import VideoCLIP_XL_v2

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

        self.index.add(vectors)
        self.names.extend(names)
        self.save()

    def remove(self, names: list[str]):
        for name in names:
            if name not in self.names:
                continue
            
            idx = self.names.index(name)
            self.index.remove_ids(np.array([idx]))
            self.names.pop(idx)

        self.save()

    def search(self, vectors: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if vectors.ndim != 2 or vectors.shape[1] != self.index.d:
            raise ValueError("Vectors must be a 2D array with shape (n, dim).")
        
        distances, indices = self.index.search(vectors, k)
        results = []
        for i in range(len(vectors)):
            result = [(self.names[idx], distances[i][j].item()) for j, idx in enumerate(indices[i]) if idx >= 0]
            results.append(result)
        return results
    
    def get(self, name: str) -> np.ndarray:
        if name not in self.names:
            return None
        
        idx = self.names.index(name)
        return self.index.reconstruct(idx)
    
    def save(self):
        faiss.write_index(self.index, f"indexes/{self.name}.index")
        with open(f"indexes/{self.name}.names", 'wb') as f:
            for name in self.names:
                f.write(struct.pack('H', len(name)))
                f.write(name.encode('utf-8'))

    def clear(self):
        self.index.reset()
        self.names.clear()

        if os.path.exists(f"indexes/{self.name}.index"):
            os.remove(f"indexes/{self.name}.index")

        if os.path.exists(f"indexes/{self.name}.names"):
            os.remove(f"indexes/{self.name}.names")

class Model:
    def __init__(self, name: str, model: BaseModel, dim: int, res: tuple[int, int], fcount: int):
        self.name = name
        self.model = model
        self.dim = dim
        self.res = res
        self.fcount = fcount

MODELS = [
    Model("VideoCLIP-XL-v2", VideoCLIP_XL_v2(), 768, (224, 224), 8),
]

class CLIPQueueProcessor:
    def __init__(self):
        self.vqueues: defaultdict[int, list] = defaultdict(list)
        self.tqueues: defaultdict[int, list] = defaultdict(list)
        self.lock = Lock()
        self.event = Event()
        self.sleeping = True
        self.event.set()

    def _add(self, model_indexes: list[int], callbacks: list, data, video: bool):
        if len(model_indexes) == 0:
            return
        
        for model_index in model_indexes:
            if model_index < 0 or model_index >= len(MODELS):
                raise IndexError("Model index out of range.")
            
        with self.lock:
            if self.sleeping:
                self.event.set()
                self.sleeping = False
            for i, model_index in enumerate(model_indexes):
                if video:
                    self.vqueues[model_index].append((callbacks[i], data))
                else:
                    self.tqueues[model_index].append((callbacks[i], data))

    def addv(self, model_indexes: list[int], callbacks: list, data: np.ndarray):
        self._add(model_indexes, callbacks, data, video=True)

    def addt(self, model_indexes: list[int], callbacks: list, data: str):
        self._add(model_indexes, callbacks, data, video=False)

    def worker(self, video: bool):
        queues = self.vqueues if video else self.tqueues
        while True:
            if self.sleeping:
                self.event.wait()
            while True:
                with self.lock:
                    if len(queues) == 0:
                        self.sleeping = True
                        self.event.clear()
                        break
                    model_index, items = queues.popitem()

                callbacks = []
                inputs = []
                for callback, _input in items:
                    callbacks.append(callback)
                    inputs.append(_input)

                if video:
                    for i, vector in enumerate(MODELS[model_index].model.process_videos(inputs)):
                        callbacks[i](vector)
                else:
                    for i, vector in enumerate(MODELS[model_index].model.process_texts(inputs)):
                        callbacks[i](vector)

class UserIndexStore:
    def __init__(self, user_key: str):
        self.indexes: list[Index] = []
        self.pending: dict[str, set[int]] = {}
        for i, model in enumerate(MODELS):
            self.indexes.append(Index(model.dim, f"{user_key}_{i}"))

    def set_pending(self, name: str, model_indexes: set[int]):
        if name in self.pending:
            self.pending[name].update(model_indexes)
        else:
            self.pending[name] = model_indexes

    def status(self, name: str) -> tuple[set[int], set[int], set[int]]:
        processing = self.pending.get(name, set())
        completed = set()
        missing = set()
        for i in set(range(len(MODELS))) - processing:
            if self.indexes[i].get(name) is not None:
                completed.add(i)
            else:
                missing.add(i)
        return missing, processing, completed

    def add(self, model_index: int, name: str, vector: np.ndarray):
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        self.indexes[model_index].add([name], np.array([vector]))
        if name in self.pending: self.pending[name].remove(model_index)

    def get(self, model_index: int, name: str) -> np.ndarray:
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        return self.indexes[model_index].get(name)

    def search(self, model_index: int, vectors: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        if model_index < 0 or model_index >= len(self.indexes):
            raise IndexError("Model index out of range.")
        
        return self.indexes[model_index].search(vectors, k)

class IndexDestination:
    def __init__(self, name: str, user_index: UserIndexStore):
        self.name = name
        self.user_index = user_index

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
        def __init__(self, r: requests.Response, media_src: str, destinations: list[IndexDestination]):
            self.r = r
            self.media_src = media_src
            self.destinations = destinations
            self.completed_model_indexes = set()
            self.lock = Lock()

    class TJob:
        def __init__(self, text: str):
            self.text = text
            self.lock = Lock()
            self.events = [Event() for _ in range(len(MODELS))]
            self.outputs: list[np.ndarray] = [None] * len(MODELS)

    def __init__(self):
        self.in_progress_v_lock = Lock()
        self.in_progress_v: dict[str, Indexer.VJob] = {}

        self.in_progress_t_lock = Lock()
        self.in_progress_t: dict[str, Indexer.TJob] = {}

        self.global_v_index = UserIndexStore("global_v")
        self.global_t_index = UserIndexStore("global_t")

        self.clip_queue_processor = CLIPQueueProcessor()
        Thread(target=self.clip_queue_processor.worker, args=(True,), daemon=True).start()
        Thread(target=self.clip_queue_processor.worker, args=(False,), daemon=True).start()

    def _try_add_destination(self, media_src: str, destination: IndexDestination) -> bool:
        if media_src in self.in_progress_v:
            job = self.in_progress_v[media_src]
            with job.lock:
                job.destinations.append(destination)
                destination.user_index.set_pending(destination.name, set(range(len(MODELS))) - job.completed_model_indexes)
                for idx in job.completed_model_indexes:
                    destination.user_index.add(idx, destination.name, self.global_v_index.get(idx, media_src))
            return True
        vec = self.global_v_index.get(0, media_src)
        if vec is not None:
            destination.user_index.add(0, destination.name, vec)
            for idx in range(1, len(MODELS)):
                destination.user_index.add(idx, destination.name, self.global_v_index.get(idx, media_src))
            return True
        return False

    def try_add_destination(self, media_src: str, destination: IndexDestination) -> bool:
        with self.in_progress_v_lock:
            return self._try_add_destination(media_src, destination)

    def resize_media(self, data: BytesIO, content_type: str, nframes: int, resolution: tuple[int, int]) -> list[np.ndarray]:
        frames = []
        if content_type in ("image/gif", "image/png"):
            image = Image.open(data)
            if content_type == "image/gif":
                if image.n_frames < nframes:
                    for i in range(image.n_frames):
                        image.seek(i)
                        frames.append(np.array(image.convert("RGB").resize(resolution)))
                    frames.extend([frames[-1]] * (nframes - len(frames)))
                else:
                    step = image.n_frames / nframes
                    for i in range(nframes):
                        image.seek(int(i * step))
                        frames.append(np.array(image.convert("RGB").resize(resolution)))
            else:
                frames = [np.array(image.convert("RGB").resize(resolution))] * nframes
        else:
            video = cv2.VideoCapture(source=data, apiPreference=cv2.CAP_FFMPEG, params=[])
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < nframes:
                for _ in range(total_frames):
                    ret, frame = video.read()
                    if not ret:
                        break
                    frames.append(cv2.resize(frame, resolution))
                frames.extend([frames[-1]] * (nframes - len(frames)))
            else:
                step = total_frames / nframes
                for i in range(nframes):
                    video.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
                    ret, frame = video.read()
                    if not ret:
                        break
                    frames.append(cv2.resize(frame, resolution))
        return frames

    def vclip_complete_callback(self, job: VJob, model_index: int, vector: np.ndarray):
        with job.lock:
            job.completed_model_indexes.add(model_index)
            for destination in job.destinations:
                destination.user_index.add(model_index, destination.name, vector)
            if len(job.completed_model_indexes) == len(MODELS):
                with self.in_progress_v_lock:
                    del self.in_progress_v[job.media_src]

    def _process(self, job: VJob):
        content_type = job.r.headers["content-type"]

        data = BytesIO()
        for chunk in job.r.iter_content(chunk_size=8192):
            data.write(chunk)
        data.seek(0)

        job.r.close()

        processed_media = {}
        grouped_models = defaultdict(list)
        for i, model in enumerate(MODELS):
            req = (model.fcount, model.res)
            if req not in processed_media:
                processed_media[req] = self.resize_media(data, content_type, *req)
                grouped_models[req].append((i, lambda v, job=job, i=i: self.vclip_complete_callback(job, i, v)))

        data.close()

        for res in grouped_models:
            model_indexes = []
            callbacks = []
            for model_index, callback in grouped_models[res]:
                model_indexes.append(model_index)
                callbacks.append(callback)
            self.clip_queue_processor.addv(model_indexes, callbacks, processed_media[res])

    def process_media(self, media_src: str, r: requests.Response, destination: IndexDestination):
        with self.in_progress_v_lock:
            if self._try_add_destination(media_src, destination):
                return

            job = Indexer.VJob(r, media_src, [IndexDestination(media_src, self.global_v_index), destination])
            self.in_progress_v[media_src] = job
        destination.user_index.set_pending(destination.name, set(range(len(MODELS))))
        Thread(target=self._process, args=(job,), daemon=True).start()

    def tclip_complete_callback(self, job: TJob, model_index: int, vector: np.ndarray):
        with self.in_progress_t_lock:
            del self.in_progress_t[job.text]
            self.global_t_index.add(model_index, job.text, vector)
        job.outputs[model_index] = vector
        job.events[model_index].set()

    def get_text_vector(self, text: str):
        with self.in_progress_t_lock:
            if text in self.in_progress_t:
                return self.in_progress_t[text]
            vec = self.global_t_index.get(0, text)
            if vec is not None:
                vecs = [vec]
                for i in range(1, len(MODELS)):
                    vecs.append(self.global_t_index.get(i, text))
                return vecs
            job = Indexer.TJob(text)
            self.in_progress_t[text] = job
        self.clip_queue_processor.addt(list(range(len(MODELS))), [lambda vec, job=job, i=i: self.tclip_complete_callback(job, i, vec) for i in range(len(MODELS))], text)
        return job

indexer = Indexer()

app = Flask(__name__)

def error(message: str):
    return jsonify({"status": "error", "message": message})

@app.route('/<user_key>/index', methods=['POST'])
def index(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    payload = request.get_json()
    if "media_src" not in payload or "name" not in payload:
        return error("Missing required fields"), 400

    media_src = payload["media_src"]
    if len(media_src) > 2000:
        return error("media_src exceeds 2000 characters"), 400

    if media_src.startswith("http://"): domain_offset = 7
    elif media_src.startswith("https://"): domain_offset = 8
    else: return error("Invalid media_src"), 400

    path_idx = media_src.find('/', domain_offset)
    if path_idx == -1:
        path_idx = len(media_src)

    domain = media_src[domain_offset:path_idx]

    if not domain or len(domain) > 256 or (not domain.endswith(".discordapp.net") and not domain == "media.tenor.co"):
        return error("Invalid domain in media_src"), 400
    
    user_index = get_user_index(user_key)
    if indexer.try_add_destination(media_src, IndexDestination(payload["name"], user_index)):
        return jsonify({"status": "ok", "message": "Indexing in progress"}), 202

    r = requests.get(media_src, stream=True)
    if "content-length" not in r.headers or "content-type" not in r.headers:
        return error("Invalid media_src"), 400

    if int(r.headers["content-length"]) > 50 * 1024 * 1024:
        return error("media_src exceeds 50MB"), 400

    if r.headers["content-type"] not in ("image/gif", "image/png", "video/mp4"):
        return error("Unsupported media type"), 400

    indexer.process_media(media_src, r, IndexDestination(payload["name"], user_index))
    return jsonify({"status": "ok", "message": "Indexing in progress"}), 202

@app.route('/<user_key>/status', methods=['GET'])
def status(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    name = request.args.get('name')
    if not name:
        return error("Missing 'name' parameter"), 400

    user_index = get_user_index(user_key)
    missing, processing, completed = user_index.status(name)
    return jsonify({
        "status": "ok",
        "missing": list(missing),
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
    
    res = indexer.get_text_vector(text)
    vectors = []
    if isinstance(res, Indexer.TJob):
        for i in range(len(MODELS)):
            res.events[i].wait()
            vectors.append(res.outputs[i])
    else:
        vectors = res

    return jsonify({
        "status": "ok",
        "results": {i: get_user_index(user_key).search(i, np.array([vectors[i]]), k=10)[0] for i in range(len(MODELS))}
    }), 200


app.run(host='127.0.0.1', port=5002, debug=True, threaded=False)
