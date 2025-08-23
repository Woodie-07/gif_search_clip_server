from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import requests
from threading import Thread, Lock, Event
from io import BytesIO
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from models import MODELS
from index import UserIndexStore, IndexDestination, State
from queue_workers import VCLIPQueueProcessor, TCLIPQueueProcessor
from resize import resize_media
import json

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
                    processed = resize_media(data, content_type, *req)
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
        self.download_executor.submit(self._download_media, url, media_src)

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

app.run(host='127.0.0.1', port=5002, debug=False, threaded=True)
