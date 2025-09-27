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

def strip_url_params(url: str) -> str:
    return url.split('?', 1)[0]

class ClientUpdate:
    pass

class StatusUpdate(ClientUpdate):
    def __init__(self, url: str, model: str, status_update: JobStatus):
        self.url = url
        self.model = model
        self.status_update = status_update

class VCLIPResult(ClientUpdate):
    def __init__(self, url: str, model: str, vector: np.ndarray):
        self.url = url
        self.model = model
        self.vector = vector

class TCLIPResult(ClientUpdate):
    def __init__(self, text: str, model: str, vector: np.ndarray):
        self.text = text
        self.model = model
        self.vector = vector

class Worker:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer

        self.v_model_queue_lengths: dict[str, int] = {}
        self.t_model_queue_lengths: dict[str, int] = {}

        self.vpending: dict[int, tuple[str, str]] = {}
        self.tpending: dict[int, tuple[str, str]] = {}

        self.pong_recv_event = None

        self.vtask_counter = 0
        self.ttask_counter = 0

        self.inference_cost = None

    async def send(self, data):
        self.writer.write(data)
        await self.writer.drain()

    async def initialise(self) -> asyncio.Task:
        data = await self.reader.readexactly(6)
        packet_id, nmodels, self.inference_cost = struct.unpack("!BBI", data)
        if packet_id != 0x00: raise ValueError("unexected packet id in initialise", packet_id)
        for _ in range(nmodels):
            name_length = await self.reader.readexactly(1)
            model_name = await self.reader.readexactly(name_length[0])
            self.t_model_queue_lengths[model_name.decode("utf-8")] = self.v_model_queue_lengths[model_name.decode("utf-8")] = 0

        t = asyncio.create_task(self.pinger())
        t.add_done_callback(lambda _, task=asyncio.current_task(): task.get_loop().create_task(self.pinger_exited(task)))
        return t

    def _get_avail_models(self, models: set[str], queues: dict[str, int]) -> tuple[set[str], int]:
        avail = set(queues.keys()) & models
        return avail, sum(queues[m] for m in avail) * self.inference_cost

    def get_avail_v_models(self, models: set[str]) -> tuple[set[str], int]:
        return self._get_avail_models(models, self.v_model_queue_lengths)

    def get_avail_t_models(self, models: set[str]) -> tuple[set[str], int]:
        return self._get_avail_models(models, self.t_model_queue_lengths)

    async def pinger_exited(self, main_task: asyncio.Task):
        self.writer.close()
        main_task.cancel()

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

    async def read_id_vector(self) -> tuple[int, np.ndarray]:
        data = await self.reader.readexactly(16)
        id, vector_len = struct.unpack("!QQ", data)
        vector_data = await self.reader.readexactly(vector_len)
        vector_data = BytesIO(vector_data)
        vector = np.load(vector_data)
        return id, vector

    async def read_packets(self) -> AsyncIterator[ClientUpdate]:
        while True:
            #print("reading packet")
            data = await self.reader.readexactly(1)
            packet_id = data[0]
            #print(f"got packet {packet_id}")
            match packet_id:
                case 0x00:
                    raise ValueError("unexpected initialise packet during main read")
                case 0x01:
                    if self.pong_recv_event is None:
                        raise ValueError("received unexpected pong packet")
                    self.pong_recv_event.set()
                case 0x02:
                    data = await self.reader.readexactly(9)
                    id, status = struct.unpack("!QB", data)
                    status = JobStatus.FAILED if status == 0 else JobStatus.PROCESSING 
                    if status == JobStatus.FAILED:
                        d = self.vpending.pop(id)
                        self.v_model_queue_lengths[d[1]] -= 1
                    else:
                        d = self.vpending[id]
                    print(f"got job {id} status update, new status {status.name}")
                    yield StatusUpdate(*d, status)
                case 0x03:
                    id, vector = await self.read_id_vector()
                    print(f"vclip job {id} complete, received vector of shape {vector.shape}")
                    d = self.vpending.pop(id)
                    self.v_model_queue_lengths[d[1]] -= 1
                    yield VCLIPResult(*d, vector)
                case 0x04:
                    id, vector = await self.read_id_vector()
                    print(f"tclip job {id} complete, received vector of shape {vector.shape}")
                    d = self.tpending.pop(id)
                    self.t_model_queue_lengths[d[1]] -= 1
                    yield TCLIPResult(*d, vector)
                case _:
                    raise ValueError("unknown packet ID", packet_id)
                
    async def submit_vclip_task(self, url: str, stripped: str, models: set[str]):
        data = struct.pack("!BQI", 0x01, self.vtask_counter, len(url)) + url.encode("utf-8") + bytes([len(models)])
        for i, model in enumerate(models):
            self.vpending[self.vtask_counter + i] = (stripped, model)
            for i, key in enumerate(self.v_model_queue_lengths):
                if key == model:
                    data += bytes([i])
                    self.v_model_queue_lengths[key] += 1
                    break
            else:
                assert False
        self.vtask_counter += len(models)
        await self.send(data)

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
        self.workers: set[Worker] = set()
        self.workers_lock = threading.Lock()
        self.event_loop: asyncio.AbstractEventLoop = None

        self.global_v_index = UserIndexStore("global_v")
        self.global_t_index = UserIndexStore("global_t")
        
        self.pending_v: dict[str, dict[str, StateContainer]] = {}
        self.pending_v_lock = threading.Lock()
        self.pending_t: defaultdict[str, dict[str, tuple[Future, set[str]]]] = defaultdict(dict)
        self.pending_t_lock = threading.Lock()

        self.failed_urls = set()

    async def run(self):
        server = await asyncio.start_server(self.handle_client, '0.0.0.0', 30456)
        self.event_loop = asyncio.get_event_loop()
        print("listening for workers")
        async with server:
            await server.serve_forever()

    def pop_pending_v(self, url: str, model: str) -> StateContainer:
        with self.pending_v_lock:
            sc = self.pending_v[url].pop(model)
            if not self.pending_v[url]:
                self.pending_v.pop(url)
            return sc
    
    def pop_pending_t(self, text: str, model: str) -> Future:
        with self.pending_t_lock:
            f, _ = self.pending_t[text].pop(model)
            if not self.pending_t[text]:
                self.pending_t.pop(text)
            return f

    async def handle_client(self, *args):
        worker = Worker(*args)
        pinger = None
        try:
            pinger = await worker.initialise()
            with self.workers_lock:
                self.workers.add(worker)

            print("worker connected")
            
            async for packet in worker.read_packets():
                if isinstance(packet, StatusUpdate):
                    sc = self.pending_v[packet.url][packet.model]
                    sc.set_state(packet.status_update)
                    if packet.status_update == JobStatus.FAILED:
                        self.pop_pending_v(packet.url, packet.model)
                        self.failed_urls.add(sc.get_data())
                elif isinstance(packet, VCLIPResult):
                    self.global_v_index.add(packet.url, packet.model, packet.vector)
                    self.pop_pending_v(packet.url, packet.model).index(packet.model, packet.vector)
                elif isinstance(packet, TCLIPResult):
                    self.global_t_index.add(packet.text, packet.model, packet.vector)
                    self.pop_pending_t(packet.text, packet.model).set_result(packet.vector)
                else:
                    assert False
        except Exception as e:
            print(f"worker disconnect: {e}")
            traceback.print_exc()
        finally:
            print("worker cleanup")
            if pinger: pinger.cancel()
            with self.workers_lock:
                self.workers.discard(worker)

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
                    await self._index(dest.user_index, dest.name, models, url)

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

    models = set(request.args.get("models").split(","))

    # return top 10 results for each model
    return jsonify({
        "status": "ok",
        "results": {model: get_user_index(user_key).search(model, np.array([vector]), k=10)[0] for model, vector in server.get_text_vectors(text, models, user_key) if vector is not None}
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
