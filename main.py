from flask import Flask, request, jsonify, Response
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

STATS_ALLOWED_IP = "127.0.0.1" # IP address allowed to fetch /metrics

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

    # in progress video CLIP job
    class VJob:
        def __init__(self, model_index: int, media_src: str, destinations: set[IndexDestination]):
            self.model_index = model_index
            self.media_src = media_src

            # if another user tries to index the same gif, their index will be added here to 'subscribe' to the result
            self.destinations = destinations
            self.lock = Lock()

    # in progress text CLIP job
    class TJob:
        def __init__(self, model_index: int, text: str):
            self.model_index = model_index
            self.text = text
            self.lock = Lock()
            self.event = Event()
            self.output = None
            self.users: set[str] = set()

        def add_user(self, user_key: str):
            self.users.add(user_key)

    def __init__(self):
        self.in_progress_v_lock = Lock()
        # stores (model_idx, media_src) -> job mappings
        self.in_progress_v: dict[tuple[int, str], Indexer.VJob] = {}
        # stores media_src -> (model_idx, destinations)s mappings 
        self.in_progress_v_dl: dict[str, defaultdict[int, set[IndexDestination]]] = {}

        self.in_progress_t_lock = Lock()
        # stores (model_idx, text) -> job mappings
        self.in_progress_t: dict[tuple[int, str], Indexer.TJob] = {}

        # whenever anything is indexed by CLIP, a copy is forever stored in the global index
        # then it can be accessed quickly by any user, no reprocessing required
        self.global_v_index = UserIndexStore("global_v")
        self.global_t_index = UserIndexStore("global_t")

        # URLs that couldn't be downloaded or preprocessed, these will be ignored
        # preprocessing is decoding the media into raw frames of the desired resolution and count
        # currently is not saved to disk
        self.failed_media_dl = set()

        # workers that perform downloading and preprocessing
        self.download_executor = ThreadPoolExecutor(max_workers=5)

        # workers that run the CLIP models
        # each worker will cycle through the required models, processing a batch of videos/texts
        self.vclip_queue_processor = VCLIPQueueProcessor()
        self.tclip_queue_processor = TCLIPQueueProcessor()
        Thread(target=self.vclip_queue_processor.worker, daemon=True).start()
        Thread(target=self.tclip_queue_processor.worker, daemon=True).start()

    def tclip_complete_callback(self, job: TJob, vector: np.ndarray):
        # CLIP has completed on a string. move it out of in progress and into the global index
        with self.in_progress_t_lock:
            del self.in_progress_t[(job.model_index, job.text)]
            self.global_t_index.add(job.model_index, job.text, vector)
        # write the CLIP vector to the job's output and set the event to wake waiting users
        job.output = vector
        job.event.set()

    def get_text_vectors(self, models: list[int], text: str, user_key: str) -> list[TJob] | list[np.ndarray]:
        vectors: list[Indexer.TJob] | list[np.ndarray] = [None] * len(models)
        has_vector_idxes = set()
        with self.in_progress_t_lock:
            for i, model_idx in enumerate(models):
                ret = self.in_progress_t.get((model_idx, text))
                if ret is not None:
                    # if this text and model is already in progress, put the job in vectors so caller can 'subscribe' to it
                    vectors[i] = ret
                    continue

                vec = self.global_t_index.get(model_idx, text)
                if vec is not None:
                    # if this text and model is already indexed, put the existing vector in vectors
                    vectors[i] = vec
                    has_vector_idxes.add(i)
                    continue

                # make a new job and queue it
                job = Indexer.TJob(model_idx, text)
                job.add_user(user_key)
                self.in_progress_t[(model_idx, text)] = job
                self.tclip_queue_processor.add(model_idx, lambda vec, job=job: self.tclip_complete_callback(job, vec), text)
                vectors[i] = job
        
        if len(has_vector_idxes) != len(models):
            # if we don't already have all vectors, convert known vectors into pseudo-jobs
            # this way the function returns either all jobs or all vectors
            for i in has_vector_idxes:
                job = Indexer.TJob(models[i], text)
                job.output = vectors[i]
                job.event.set()
                vectors[i] = job

        return vectors

    def vclip_complete_callback(self, job: VJob, model_index: int, vector: np.ndarray):
        # CLIP completed on a video, add to global index and all subscribed user indexes
        with self.in_progress_v_lock:
            del self.in_progress_v[(model_index, job.media_src)]
            self.global_v_index.add(model_index, job.media_src, vector)
            for destination in job.destinations:
                with destination.lock:
                    # used for cancellation
                    # if a user removed the GIF while it's processing, the cancel function will wait on the lock
                    # once the lock is acquired, it checks this event. if set, it knows it's too late and must remove from the user's index
                    # if not set, it sets it, preventing the vector from being added to the user's index at all
                    if destination.event.is_set():
                        continue
                    destination.user_index.add(model_index, destination.name, vector)
                    destination.event.set()

    def _download_media(self, url: str, media_src: str):
        failed = True
        required_models = self.in_progress_v_dl[media_src]
        popped = False
        try:
            # stream so we don't download huge media before we know what the size is
            r = requests.get(url, stream=True)
            if "content-length" not in r.headers or "content-type" not in r.headers:
                return

            if int(r.headers["content-length"]) > 50 * 1024 * 1024: # ignore over 50 MB
                return

            content_type = r.headers["content-type"]
            if content_type not in ("image/gif", "image/png", "video/mp4"): # ignore unknown file types
                return
            
            data = BytesIO(r.content)

            # we just moved the content into a BytesIO, r can be deleted to save a little memory
            del r 

            jobs = []
            with self.in_progress_v_lock:
                # download complete, let's remove from in progress dl and create per-model jobs
                # prevents new model indexes from being added to this download job as we're about to process those
                del self.in_progress_v_dl[media_src]
                popped = True
                for model_idx, destinations in required_models.items():
                    job = Indexer.VJob(model_idx, media_src, destinations)
                    self.in_progress_v[(model_idx, media_src)] = job
                    jobs.append(job)

            processed_media = {}
            for i, (model_idx, destinations) in enumerate(required_models.items()):
                # get the media requirements for this model and preprocess to match them, if not already done
                model = MODELS[model_idx]
                req = (model.fcount, model.res, model.resize_mode)
                if req not in processed_media:
                    processed = resize_media(data, content_type, *req)
                    data.seek(0)
                    if not processed:
                        return
                    processed_media[req] = processed
                job = jobs[i]
                # queue video CLIP job! runs the lambda when it's done.
                self.vclip_queue_processor.add(model_idx, lambda vector, job=job, model_idx=model_idx: self.vclip_complete_callback(job, model_idx, vector), processed_media[req])
            failed = False
        finally:
            with self.in_progress_v_lock:
                # if something went wrong during download/preprocess, mark this media as failed
                # it's entirely possible that it failed because the Discord rotating CDN args are now invalid because we took too long if long backlog
                # that's why we don't mark media_src (URL with stripped params) as bad, rather the full URL
                if failed: self.failed_media_dl.add(url)
                for model_idx, destinations in required_models.items():
                    for destination in destinations:
                        # update user statuses
                        destination.user_index.set_state(destination.name, model_idx, State.FAILED if failed else State.PROCESSING, destinations)
                # if we failed before deleting, do it now
                if not popped: del self.in_progress_v_dl[media_src]

    def enqueue(self, user_index: UserIndexStore, media_src: str, url: str, name: str, models: set[int]):
        # we need to find what stage the media is in so we can subscribe to the pending result if already queued
        # therefore, we acquire the lock so the media won't move between stages as we're looking for it so we miss it
        with self.in_progress_v_lock:
            models = set(filter(lambda m: user_index.get(m, name) is None, models))
            if not models: return

            # is in global? copy to user
            def from_global_filter(model_index: int) -> bool:
                res = self.global_v_index.get(model_index, media_src)
                if res is not None:
                    user_index.add(model_index, name, res)
                    return False
                return True
            models = set(filter(from_global_filter, models))
            if not models: return

            # is CLIP in progress? subscribe to it
            def subscribe_to_embedding_filter(model_index: int) -> bool:
                vjob = self.in_progress_v.get((model_index, media_src))
                if vjob is None:
                    return True
                vjob.destinations.add(IndexDestination(name, user_index))
                user_index.set_state(name, model_index, State.PROCESSING, vjob.destinations)
                return False
            models = set(filter(subscribe_to_embedding_filter, models))
            if not models: return

            # is failed? mark as failed and ignore
            if url in self.failed_media_dl:
                for model_index in models:
                    user_index.set_state(name, model_index, State.FAILED)
                return
            
            # is downloading? subscribe to it
            downloading = self.in_progress_v_dl.get(media_src)
            if downloading is not None:
                for model_index in models:
                    downloading[model_index].add(IndexDestination(name, user_index))
                    user_index.set_state(name, model_index, State.DOWNLOADING, downloading[model_index])
                return
            
            # it's had no prior indexing attempt. begin indexing
            self.in_progress_v_dl[media_src] = defaultdict(set)
            for model_index in models:
                self.in_progress_v_dl[media_src][model_index].add(IndexDestination(name, user_index))
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
        if len(name) > 2048:
            return error("Each name must not exceed 2048 characters"), 400

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
        # strip url params from url to make media_src
        # this way we don't treat different Discord CDN params as a different image
        params_idx = url.find("?")
        if params_idx != -1:
            media_src = url[:params_idx]
        else:
            media_src = url
        indexer.enqueue(user_index, media_src, url, names[i], models)

    # remove any GIFs in the user index (or pending) that aren't in this index request
    # therefore we delete GIFs that the user unfavourites
    user_index.remove_not_present(set(names))

    return jsonify({"status": "ok", "message": "Indexing started"}), 202

@app.route('/<user_key>/status', methods=['GET'])
def status(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    name = request.args.get('name')
    if not name:
        return error("Missing 'name' parameter"), 400

    # get the status of each model index for a given GIF
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
    
    # this function either returns all TJobs or all vectors
    # if TJobs, wait for each event and fetch output, else use the vectors directly
    # currently we run the text through all models, not just user-selected ones
    res = indexer.get_text_vectors(list(range(len(MODELS))), text, user_key)
    vectors = []
    if isinstance(res[0], Indexer.TJob):
        for i in range(len(MODELS)):
            res[i].event.wait()
            vectors.append(res[i].output)
    else:
        vectors = res

    # return top 10 results for each model
    return jsonify({
        "status": "ok",
        "results": {i: get_user_index(user_key).search(i, np.array([vectors[i]]), k=10)[0] for i in range(len(MODELS))}
        }), 200

# prometheus format metrics
@app.route('/metrics', methods=['GET'])
def metrics():
    ip = request.remote_addr
    if ip != STATS_ALLOWED_IP:
        return "disallowed"
    
    stats = ""
    stats += f"loaded_user_indexes {len(user_indexes)}\n"
    stats += f"failed_media_dl {len(indexer.failed_media_dl)}\n"

    for i, index in enumerate(indexer.global_v_index.indexes):
        stats += "indexed_videos{model_idx=\"" + str(i) + "\"} " + str(len(index.names)) + "\n"
    for i, index in enumerate(indexer.global_t_index.indexes):
        stats += "indexed_text{model_idx=\"" + str(i) + "\"} " + str(len(index.names)) + "\n"

    stats += f"in_progress_text {len(indexer.in_progress_t)}\n"
    stats += f"in_progress_video_downloads {len(indexer.in_progress_v_dl)}\n"
    stats += f"in_progress_video_clip {len(indexer.in_progress_v)}\n"

    per_user_text = defaultdict(int)
    for tjob in indexer.in_progress_t.values():
        for user_key in tjob.users:
            per_user_text[user_key] += 1
    for user_key, count in per_user_text.items():
        stats += "in_progress_text_per_user{user_key=\"" + user_key + "\"} " + str(count) + "\n"
    
    per_user_v_dl = defaultdict(int)
    for models in indexer.in_progress_v_dl.values():
        for user_key in {index_dest.user_index.user_key for index_dests in models.values() for index_dest in index_dests}:
            per_user_v_dl[user_key] += 1
    for user_key, count in per_user_v_dl.items():
        stats += "in_progress_video_downloads_per_user{user_key=\"" + user_key + "\"} " + str(count) + "\n"

    per_user_v = defaultdict(int)
    for vjob in indexer.in_progress_v.values():
        for user_key in {index_dest.user_index.user_key for index_dest in vjob.destinations}:
            per_user_v[user_key] += 1
    for user_key, count in per_user_v.items():
        stats += "in_progress_video_clip_per_user{user_key=\"" + user_key + "\"} " + str(count) + "\n"

    return Response(stats, mimetype="text/plain")
