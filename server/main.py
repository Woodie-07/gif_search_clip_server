import asyncio
import threading
from typing import AsyncIterator, Iterator
from collections import defaultdict
from enum import IntEnum
import struct
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from concurrent.futures import Future
import traceback

from index import UserIndexStore, StateContainer, JobStatus, IndexDestination

STATS_ALLOWED_IP = "127.0.0.1" # IP address allowed to fetch /metrics

# removes question mark and everything after it from a URL
def strip_url_params(url: str) -> str:
    return url.split('?', 1)[0]

# container classes for updates sent from workers
class ClientUpdate:
    pass

class StatusUpdate(ClientUpdate):
    def __init__(self, url: str, model: str, status_update: JobStatus):
        self.url = url # gif download url, stripped
        self.model = model # model name of this task
        self.status_update = status_update # enum representing new status, e.g. processing, completed, or failed

class VCLIPResult(ClientUpdate):
    def __init__(self, url: str, model: str, vector: np.ndarray):
        self.url = url
        self.model = model
        self.vector = vector # numpy array of the vector from the clip model sent by the worker

# same here but for a text query rather than GIF
class TCLIPResult(ClientUpdate):
    def __init__(self, text: str, model: str, vector: np.ndarray):
        self.text = text
        self.model = model
        self.vector = vector

class Worker:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        # handles to CLIP worker connection streams
        self.reader = reader
        self.writer = writer

        # model name -> number of queued tasks for that model
        # used when picking which worker to send tasks to (lowest queue length preferred)
        self.v_model_queue_lengths: dict[str, int] = {}
        self.t_model_queue_lengths: dict[str, int] = {}

        # numerical id (simple counter) -> (stripped url/text, model name)
        self.vpending: dict[int, tuple[str, str]] = {}
        self.tpending: dict[int, tuple[str, str]] = {}

        # when a ping is sent, this event is created and waited upon
        # if timeout reached, means we didn't get pong from worker so we consider it dead
        self.pong_recv_event = None

        # simple counters for generating unique task ids
        self.vtask_counter = 0
        self.ttask_counter = 0

        # worker's reported cost for inference, higher = slower so less preferred
        # worker queue length is multiplied by this when picking best worker
        self.inference_cost = None

    # helper func to send data immediately
    async def send(self, data):
        self.writer.write(data)
        await self.writer.drain()

    # worker just connected, now we need to read initial info from the worker
    async def initialise(self) -> asyncio.Task:
        data = await self.reader.readexactly(6) # 1 byte packet id (should be 0x00), 1 byte nmodels, 4 byte inference cost big-endian, all unsigned
        packet_id, nmodels, self.inference_cost = struct.unpack("!BBI", data)

        if packet_id != 0x00: raise ValueError("unexected packet id in initialise", packet_id)

        # read worker's reported supported models
        for _ in range(nmodels):
            name_length = await self.reader.readexactly(1)
            model_name = await self.reader.readexactly(name_length[0])
            self.t_model_queue_lengths[model_name.decode("utf-8")] = self.v_model_queue_lengths[model_name.decode("utf-8")] = 0

        # begin sending periodic ping packets to worker to ensure it's still alive
        t = asyncio.create_task(self.pinger())
        # once the pinger exits (worker dead), call pinger_exited with the current task (will be main recv loop)
        t.add_done_callback(lambda _, task=asyncio.current_task(): task.get_loop().create_task(self.pinger_exited(task)))
        return t

    # after pinger exits, close connection and cancel main recv loop
    async def pinger_exited(self, main_task: asyncio.Task):
        self.writer.close()
        main_task.cancel()

    # send ping packet (0x00) around every 15 secs, waiting up to 10 secs for pong (0x01) response
    # if no pong received in time, assume worker is dead and exit
    async def pinger(self) -> None:
        while True:
            self.pong_recv_event = asyncio.Event()
            await self.send(b"\x00")
            try:
                await asyncio.wait_for(self.pong_recv_event.wait(), timeout=10)
            except:
                print("worker ping timeout")
                return
            self.pong_recv_event = None
            await asyncio.sleep(15)

    # get a list of supported models for this worker and the total weighted queue length for those models
    def _get_avail_models(self, models: set[str], queues: dict[str, int]) -> tuple[set[str], int]:
        avail = set(queues.keys()) & models
        return avail, sum(queues[m] for m in avail) * self.inference_cost

    # for video CLIP
    def get_avail_v_models(self, models: set[str]) -> tuple[set[str], int]:
        return self._get_avail_models(models, self.v_model_queue_lengths)

    # for text CLIP
    def get_avail_t_models(self, models: set[str]) -> tuple[set[str], int]:
        return self._get_avail_models(models, self.t_model_queue_lengths)

    # read 8 byte job id and 8 byte vector length, then read vector data load it into a ndarray
    async def read_id_vector(self) -> tuple[int, np.ndarray]:
        data = await self.reader.readexactly(16)
        id, vector_len = struct.unpack("!QQ", data)
        vector_data = await self.reader.readexactly(vector_len)
        vector_data = BytesIO(vector_data)
        vector = np.load(vector_data)
        return id, vector

    # main recv loop, yields ClientUpdate objects as they are received
    async def read_packets(self) -> AsyncIterator[ClientUpdate]:
        while True:
            #print("reading packet")
            data = await self.reader.readexactly(1)
            packet_id = data[0]
            #print(f"got packet {packet_id}")
            match packet_id:
                case 0x00: # init
                    raise ValueError("unexpected initialise packet during main read") # 0x00 should only be sent once at start which we should've already handled
                case 0x01: # pong
                    if self.pong_recv_event is None: # we got a pong without sending a ping??
                        raise ValueError("received unexpected pong packet")
                    self.pong_recv_event.set() # alert pinger that pong recv'd
                case 0x02: # video CLIP job status update
                    data = await self.reader.readexactly(9) # 8 byte job id, 1 byte status
                    id, status = struct.unpack("!QB", data)
                    # only two possible status updates atm
                    # downloading status is assumed as soon as job is sent to worker
                    status = JobStatus.FAILED if status == 0 else JobStatus.PROCESSING
                    if status == JobStatus.FAILED: # download failure
                        # remove job id -> (url, model) mapping and decrement queue length for that model
                        d = self.vpending.pop(id)
                        self.v_model_queue_lengths[d[1]] -= 1
                    else:
                        d = self.vpending[id] # status update of job downloading -> processing
                    print(f"got job {id} status update, new status {status.name}")
                    yield StatusUpdate(*d, status)
                case 0x03: # video CLIP result vector
                    id, vector = await self.read_id_vector()
                    print(f"vclip job {id} complete, received vector of shape {vector.shape}")
                    d = self.vpending.pop(id)
                    self.v_model_queue_lengths[d[1]] -= 1
                    yield VCLIPResult(*d, vector)
                case 0x04: # text CLIP result vector
                    id, vector = await self.read_id_vector()
                    print(f"tclip job {id} complete, received vector of shape {vector.shape}")
                    d = self.tpending.pop(id)
                    self.t_model_queue_lengths[d[1]] -= 1
                    yield TCLIPResult(*d, vector)
                case _: # ??
                    raise ValueError("unknown packet ID", packet_id)
            
    # we have some video CLIP work to do. we have the GIF download URL and its URL param stripped form along with the set of models to pass it through
    async def submit_vclip_task(self, url: str, stripped: str, models: set[str]):
        # 1 byte packet id (0x01), 8 byte task id, 4 byte url length, url data, 1 byte nmodels, nmodels * 1 byte model indices
        data = struct.pack("!BQI", 0x01, self.vtask_counter, len(url)) + url.encode("utf-8") + bytes([len(models)])
        for i, model in enumerate(models):
            self.vpending[self.vtask_counter + i] = (stripped, model) # store id -> (url, model) mapping
            # find requested model's index for this worker
            # happens to be in correct order since we read them in order during initialise
            for i, key in enumerate(self.v_model_queue_lengths):
                if key == model:
                    data += bytes([i])
                    self.v_model_queue_lengths[key] += 1 # also increment queue length for that model
                    break
            else:
                assert False # we got asked to run a model this worker doesn't support??
        
        # task ids are unique per model, so if we send 3 models for one URL, they get 3 consecutive ids
        # hence, increment counter by number of models so we have no overlap next time
        self.vtask_counter += len(models)
        await self.send(data)

    # same here but for text CLIP
    async def submit_tclip_task(self, text: str, models: set[str]):
        data = struct.pack("!BQI", 0x02, self.ttask_counter, len(text)) + text.encode("utf-8") + bytes([len(models)])
        for i, model in enumerate(models):
            self.tpending[self.ttask_counter + i] = (text, model)
            for i, key in enumerate(self.t_model_queue_lengths):
                if key == model:
                    data += bytes([i])
                    self.t_model_queue_lengths[key] += 1
                    break
            else:
                assert False
        self.ttask_counter += len(models)
        await self.send(data)

class Server:
    
    def __init__(self):
        self.workers: set[Worker] = set() # stores all currently available workers
        self.workers_lock = threading.Lock() # we need this lock bc metrics thread can read the set while server task modifies it
        self.event_loop: asyncio.AbstractEventLoop = None # set in run(), handle to server's event loop used so flask thread can schedule tasks on the server thread

        # indexes for storing all vectors ever computed, used to avoid recomputing vectors for same URL/text across different users
        self.global_v_index = UserIndexStore("global_v")
        self.global_t_index = UserIndexStore("global_t")
        
        self.pending_v: dict[str, dict[str, StateContainer]] = {} # stripped url -> model name -> StateContainer
        self.pending_v_lock = threading.Lock() # read by metrics
        self.pending_t: defaultdict[str, dict[str, tuple[Future, set[str]]]] = defaultdict(dict) # text -> model name -> (Future, set of user keys waiting on this for metrics)
        self.pending_t_lock = threading.Lock() # read by metrics

        # urls that failed to download, so we don't keep retrying them
        # unstripped so if it's just e.g. discord CDN params expired, we retry next time we see different params
        self.failed_urls = set()

    # listen for incoming connections from workers and run handle_client for each
    async def run(self):
        server = await asyncio.start_server(self.handle_client, '0.0.0.0', 30456)
        self.event_loop = asyncio.get_event_loop()
        print("listening for workers")
        async with server:
            await server.serve_forever()

    # from pending_v, remove and return the StateContainer for a given url and model
    def pop_pending_v(self, url: str, model: str) -> StateContainer:
        with self.pending_v_lock:
            sc = self.pending_v[url].pop(model)
            if not self.pending_v[url]:
                self.pending_v.pop(url)
            return sc
    
    # same here but for pending_t
    def pop_pending_t(self, text: str, model: str) -> Future:
        with self.pending_t_lock:
            f, _ = self.pending_t[text].pop(model)
            if not self.pending_t[text]:
                self.pending_t.pop(text)
            return f

    # new worker!
    async def handle_client(self, *args):
        worker = Worker(*args)
        pinger = None
        try:
            pinger = await worker.initialise() # hold a handle to the pinger task so we can cancel it later if needed
            with self.workers_lock:
                self.workers.add(worker)

            print("worker connected")
            
            async for packet in worker.read_packets():
                if isinstance(packet, StatusUpdate):
                    # status update for vclip task, update StateContainer and if failed, mark url as failed
                    sc = self.pending_v[packet.url][packet.model]
                    sc.set_state(packet.status_update)
                    if packet.status_update == JobStatus.FAILED:
                        self.pop_pending_v(packet.url, packet.model)
                        self.failed_urls.add(sc.get_data())
                elif isinstance(packet, VCLIPResult):
                    # vclip complete, add vector to global index and to each user's index that requested it
                    self.global_v_index.add(packet.url, packet.model, packet.vector)
                    self.pop_pending_v(packet.url, packet.model).index(packet.model, packet.vector)
                elif isinstance(packet, TCLIPResult):
                    # same for tclip
                    self.global_t_index.add(packet.text, packet.model, packet.vector)
                    self.pop_pending_t(packet.text, packet.model).set_result(packet.vector)
                else:
                    assert False # worker yielded an unknown packet type??
        except Exception as e:
            print(f"worker disconnect: {e}")
            traceback.print_exc()
        finally:
            # worker disconnected for whatever reason
            print("worker cleanup")
            if pinger: pinger.cancel()
            with self.workers_lock:
                self.workers.discard(worker)

            # for each pending vclip task, find a new worker to send it to
            grouped: defaultdict[str, defaultdict[IndexDestination, set[str]]] = defaultdict(lambda: defaultdict(set))
            for url, model in worker.vpending.values():
                models = self.pending_v[url]
                dests = models.pop(model).get_destinations()
                if not models:
                    self.pending_v.pop(url)
                
                for dest in dests:
                    grouped[url][dest].add(model)

            for url, dests in grouped.items():
                for dest, models in dests.items():
                    await self._index(dest.user_index, dest.name, models, url) # find a new home for this task

            # now same for tclip tasks
            # text -> [(model name, future)]
            # i think this is right? i'm tired...
            grouped_t: defaultdict[str, list[tuple[str, Future]]] = defaultdict(list)
            for text, model in worker.tpending.values():
                f, _ = self.pending_t[text][model]
                grouped_t[text].append((model, f))

            for text, models in grouped_t.items():
                await self._get_text_vectors(text, models)

    def find_best_worker(self, models: set[str], is_video: bool):
        best = (set(), 0, None)
        for worker in self.workers:
            if is_video:
                avail, qlen = worker.get_avail_v_models(models)
            else:
                avail, qlen = worker.get_avail_t_models(models)
            if len(avail) > len(best[0]) or (len(avail) == len(best[0]) and qlen < best[1]):
                best = (avail, qlen, worker)

        return best

    async def _index(self, user_index: UserIndexStore, name: str, models: set[str], url: str):
        models = models.copy()
        stripped = strip_url_params(url)

        models = set(filter(lambda m: not user_index.is_present(m, name), models))
        if not models: return

        def from_global_filter(model_name: str) -> bool:
            if (vec := self.global_v_index.get(model_name, stripped)) is not None:
                user_index.add(name, model_name, vec)
                return False
            return True

        models = set(filter(from_global_filter, models))
        if not models: return

        if url in self.failed_urls:
            for model in models:
                sc = StateContainer()
                sc.set_state(JobStatus.FAILED)
                user_index.add_pending(name, model, sc)
            return

        if stripped in self.pending_v:
            pending_models = models & set(self.pending_v[stripped].keys())
            models -= pending_models
            for model in pending_models:
                self.pending_v[stripped][model].add_destination(IndexDestination(name, user_index))
                user_index.add_pending(name, model, self.pending_v[stripped][model])

        while models:
            worker_models, _, worker = self.find_best_worker(models, True)
            if worker is None:
                # ignore models not available on any worker
                break
            for model in worker_models:
                state_container = StateContainer(url)
                state_container.add_destination(IndexDestination(name, user_index))
                if stripped not in self.pending_v:
                    self.pending_v[stripped] = {}
                self.pending_v[stripped][model] = state_container
                user_index.add_pending(name, model, state_container)
            await worker.submit_vclip_task(url, stripped, worker_models)
            models -= worker_models

    def index(self, user_index: UserIndexStore, names: list[str], models: set[str], urls: list[str]):
        def create_tasks():
            for name, url in zip(names, urls):
                self.event_loop.create_task(self._index(user_index, name, models, url))

        self.event_loop.call_soon_threadsafe(create_tasks)
        self.event_loop.call_soon_threadsafe(lambda: user_index.remove_not_present(set(names)))

    async def _get_text_vectors(self, text: str, models: list[tuple[str, Future]]):
        def from_global_filter(model: tuple[str, Future]) -> bool:
            if (vec := self.global_t_index.get(model[0], text)) is not None:
                model[1].set_result(vec)
                return False
            return True

        new_models: set[str] = set()
        model_to_future: dict[str, Future] = {}
        for model, future in filter(from_global_filter, models):
            new_models.add(model)
            model_to_future[model] = future

        while new_models:
            worker_models, _, worker = self.find_best_worker(new_models, False)
            if worker is None:
                for model in new_models:
                    model_to_future[model].set_result(None)
                break
            await worker.submit_tclip_task(text, worker_models)
            new_models -= worker_models

    def get_text_vectors(self, text: str, models: set[str], user_key: str) -> Iterator[tuple[str, np.ndarray]]:
        pending: list[tuple[str, Future]] = []
        new_models: list[tuple[str, Future]] = []
        with self.pending_t_lock:
            for model in models:
                if model in self.pending_t[text]:
                    f, u = self.pending_t[text][model]
                    u.add(user_key)
                else:
                    f, _ = self.pending_t[text][model] = (Future(), {user_key})
                    new_models.append((model, f))
                pending.append((model, f))
        self.event_loop.call_soon_threadsafe(self.event_loop.create_task, self._get_text_vectors(text, new_models))

        for model_name, future in pending:
            yield model_name, future.result()

server = Server()

def run_server():
    print("running server")
    asyncio.run(server.run())
    print("server exited")

threading.Thread(target=run_server, daemon=True).start()

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

    models_list: list[str] = payload["models"]
    if not isinstance(models_list, list):
        return error("models must be a list"), 400

    if len(models_list) == 0 or any(not isinstance(m, str) for m in models_list):
        return error("invalid models"), 400

    models: set[str] = set(models_list)

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

    server.index(user_index, names, models, media_srcs)

    return jsonify({"status": "ok", "message": "Indexing started"}), 202

@app.route('/<user_key>/search', methods=['GET'])
def search(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    text = request.args.get("text")
    if not text:
        return error("Missing 'text' parameter"), 400
    
    k = request.args.get("k")
    if not k:
        k = 20
    else:
        try:
            k = int(k)
        except ValueError:
            return error("Invalid 'k' parameter"), 400

    models = set(request.args.get("models").split(","))

    # return top k results for each model
    return jsonify({
        "status": "ok",
        "results": {model: get_user_index(user_key).search(model, np.array([vector]), k=k)[0] for model, vector in server.get_text_vectors(text, models, user_key) if vector is not None}
        }), 200

@app.route('/<user_key>/status/<name>', methods=['GET'])
def status(user_key, name):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    # get the status of each model name for a given GIF
    failed, downloading, processing, completed = get_user_index(user_key).status(name)
    return jsonify({
        "status": "ok",
        "failed": list(failed),
        "downloading": list(downloading),
        "processing": list(processing),
        "completed": list(completed)
    }), 200

@app.route('/<user_key>/statuscounts', methods=['GET'])
def statuscounts(user_key):
    if not is_valid_user_key(user_key):
        return error("Invalid user key"), 400

    # get the number of GIFs at each status for each model name
    counts = get_user_index(user_key).status_counts()
    return jsonify({
        "status": "ok",
        "counts": counts
    }), 200
    

# prometheus format metrics
@app.route('/metrics', methods=['GET'])
def metrics():
    ip = request.remote_addr
    if ip != STATS_ALLOWED_IP:
        return "disallowed"
    
    stats = ""
    stats += f"loaded_user_indexes {len(user_indexes)}\n"
    stats += f"failed_media_dl {len(server.failed_urls)}\n"
    
    model_worker_counts = defaultdict(int)
    with server.workers_lock:
        for worker in server.workers:
            ip, port = worker.writer.get_extra_info('peername')
            for model in worker.v_model_queue_lengths.keys():
                model_worker_counts[model] += 1
                stats += "worker_model_vqueue_lengths{worker=\"" + ":".join([ip, str(port)]) + "\", model=\"" + model + "\"} " + str(worker.v_model_queue_lengths[model]) + "\n"
                stats += "worker_model_tqueue_lengths{worker=\"" + ":".join([ip, str(port)]) + "\", model=\"" + model + "\"} " + str(worker.t_model_queue_lengths[model]) + "\n"
    for model in model_worker_counts:
        stats += "workers_per_model{model_name=\"" + model + "\"} " + str(model_worker_counts[model]) + "\n"
    stats += f"nworkers {len(server.workers)}\n"

    for model, index in server.global_v_index.indexes.items():
        stats += "indexed_videos{model_name=\"" + model + "\"} " + str(len(index.names)) + "\n"
    for model, index in server.global_t_index.indexes.items():
        stats += "indexed_text{model_name=\"" + model + "\"} " + str(len(index.names)) + "\n"

    in_progress_t = 0
    per_user_text = defaultdict(int)
    with server.pending_t_lock:
        for models in server.pending_t.values():
            in_progress_t += len(models)
            for _, u in models.values():
                for user_key in u:
                    per_user_text[user_key] += 1
    stats += f"in_progress_text {in_progress_t}\n"
    for user_key, count in per_user_text.items():
        stats += "in_progress_text_per_user{user_key=\"" + user_key + "\"} " + str(count) + "\n"

    in_progress_v_dl = in_progress_v_clip = 0
    per_user_v_dl = defaultdict(int)
    per_user_v_clip = defaultdict(int)
    with server.pending_v_lock:
        for models in server.pending_v.values():
            for sc in models.values():
                s = sc.get_state()
                if s == JobStatus.DOWNLOADING:
                    for index_dest in sc.get_destinations():
                        per_user_v_dl[index_dest.user_index.user_key] += 1
                    in_progress_v_dl += 1
                elif s == JobStatus.PROCESSING:
                    for index_dest in sc.get_destinations():
                        per_user_v_clip[index_dest.user_index.user_key] += 1
                    in_progress_v_clip += 1
                else:
                    assert False
    stats += f"in_progress_video_downloads {in_progress_v_dl}\n"
    stats += f"in_progress_video_clip {in_progress_v_clip}\n"
    for user_key, count in per_user_v_dl.items():
        stats += "in_progress_video_downloads_per_user{user_key=\"" + user_key + "\"} " + str(count) + "\n"
    for user_key, count in per_user_v_clip.items():
        stats += "in_progress_video_clip_per_user{user_key=\"" + user_key + "\"} " + str(count) + "\n"

    return Response(stats, mimetype="text/plain")

@app.route('/models', methods=['GET'])
def models():
    all_models = set()
    with server.workers_lock:
        for worker in server.workers:
            all_models.update(set(worker.v_model_queue_lengths.keys()))

    recommended_weights = {
        "VideoCLIP-XL-v2": 0.5
    }

    return jsonify({model: recommended_weights.get(model, 0.0) for model in all_models}), 200
