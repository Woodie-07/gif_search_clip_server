import asyncio
import struct
from io import BytesIO
import requests
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict
from enum import IntEnum
from threading import Thread, Lock
import numpy as np
import traceback
from multiprocessing import Process

import config
from models import MODELS
from resize import resize_media
from queue_workers import VCLIPQueueProcessor, TCLIPQueueProcessor

assert 0 < config.INFERENCE_COST < 2**32

def strip_url_params(url: str) -> str:
    return url.split('?', 1)[0]

def download(url: str) -> BytesIO:
    # stream so we don't download huge media before we know what the size is
    r = requests.get(url, stream=True)
    if "content-length" not in r.headers or "content-type" not in r.headers:
        raise ValueError("missing headers")

    if int(r.headers["content-length"]) > 50 * 1024 * 1024: # ignore over 50 MB
        raise ValueError("file too large")

    content_type = r.headers["content-type"]
    if content_type not in ("image/gif", "image/jpeg", "image/webp", "image/png", "video/mp4"): # ignore unknown file types
        raise ValueError("unsupported file type", content_type)

    return content_type, BytesIO(r.content)

class JobStatusUpdate(IntEnum):
    FAILED = 0
    PROCESSING = 1

class Connection:
    def __init__(self):
        self.reader: asyncio.StreamReader = None
        self.writer: asyncio.StreamWriter = None

        self.downloading: dict[str, dict[int, int]] = {}
        self.downloading_lock = Lock()

        self.downloader = ThreadPoolExecutor(max_workers=5)

        self.vclip_queue_processor = VCLIPQueueProcessor()
        self.tclip_queue_processor = TCLIPQueueProcessor()

        Thread(target=self.vclip_queue_processor.worker, daemon=True).start()
        Thread(target=self.tclip_queue_processor.worker, daemon=True).start()

        self.dead = False

    async def close(self):
        self.dead = True

        self.downloader.shutdown()

        self.vclip_queue_processor.clear()
        self.tclip_queue_processor.clear()

        if self.writer: self.writer.close()

    async def send(self, data):
        self.writer.write(data)
        await self.writer.drain()

    async def _run(self):
        print("connecting to server...")
        self.reader, self.writer = await asyncio.open_connection(*config.SERVER)

        assert len(MODELS) < 256
        payload = struct.pack("!BBI", 0x00, len(MODELS), config.INFERENCE_COST)
        for model in MODELS:
            assert len(model.name) < 256
            payload += bytes([len(model.name)]) + model.name.encode("utf-8")
        await self.send(payload)

        print("connected to server!")

        while True:
            data = await self.reader.readexactly(1)
            handlers = {
                0x00: self.handle_ping,
                0x01: self.handle_vjob,
                0x02: self.handle_tjob
            }
            packet_id = data[0]
            if packet_id not in handlers:
                raise ValueError(f"unknown packet ID", packet_id)
            await handlers[packet_id]()

    async def run(self):
        try:
            await self._run()
        finally:
            await self.close()

    async def handle_ping(self):
        await self.send(b"\x01")

    async def handle_vjob(self):
        data = await self.reader.readexactly(12)
        id, url_len = struct.unpack("!QI", data)
        url = await self.reader.readexactly(url_len)
        models_count = await self.reader.readexactly(1)
        model_indexes = await self.reader.readexactly(models_count[0])

        url = url.decode("utf-8")
        stripped = strip_url_params(url)

        with self.downloading_lock:
            models = self.downloading.get(stripped, None)
            if models is None:
                future = self.downloader.submit(download, url)
                models = self.downloading[stripped] = defaultdict(set)
                event_loop = asyncio.get_event_loop()
                future.add_done_callback(lambda f, stripped=stripped: self.download_complete(stripped, f, event_loop))

            for i, idx in enumerate(model_indexes):
                assert idx not in models
                models[idx] = id + i

    async def handle_tjob(self):
        data = await self.reader.readexactly(12)
        id, text_len = struct.unpack("!QI", data)
        data = await self.reader.readexactly(text_len)
        models_count = await self.reader.readexactly(1)
        model_indexes = {byte for byte in await self.reader.readexactly(models_count[0])}

        text = data.decode("utf-8")
        event_loop = asyncio.get_event_loop()
        for i, model_idx in enumerate(model_indexes):
            self.tclip_queue_processor.add(model_idx, text, lambda vector, id=id+i: event_loop.call_soon_threadsafe(event_loop.create_task, self.tclip_complete(id, vector)))

    async def send_job_status(self, id: int, status: JobStatusUpdate):
        await self.send(struct.pack("!BQB", 0x02, id, status.value))

    def download_complete(self, stripped: str, future: Future, event_loop: asyncio.AbstractEventLoop):
        if self.dead: return

        with self.downloading_lock:
            models = self.downloading.pop(stripped)

        try:
            content_type, data = future.result()
        except Exception as e:
            print(f"setting jobs as failed due to {e}")
            for id in models.values():
                event_loop.call_soon_threadsafe(event_loop.create_task, self.send_job_status(id, JobStatusUpdate.FAILED))
            return
        
        for id in models.values():
            event_loop.call_soon_threadsafe(event_loop.create_task, self.send_job_status(id, JobStatusUpdate.PROCESSING))
        
        resized = {}
        for model_idx, id in models.items():
            model = MODELS[model_idx]
            req = (model.fcount, model.res, model.resize_mode)
            this_resized = resized.get(req, None)
            if this_resized is None:
                this_resized = resize_media(data, content_type, *req)
                data.seek(0)
                if not this_resized:
                    print("setting jobs as failed due to resize fail")
                    for id in models.values():
                        event_loop.call_soon_threadsafe(event_loop.create_task, self.send_job_status(id, JobStatusUpdate.FAILED))
                    return
                resized[req] = this_resized
            self.vclip_queue_processor.add(model_idx, this_resized, lambda vector, id=id: event_loop.call_soon_threadsafe(event_loop.create_task, self.vclip_complete(id, vector)))

    async def vclip_complete(self, id: int, vector: np.ndarray):
        print("vclip complete! sending vector")
        if self.dead: return

        serialised = BytesIO()
        np.save(serialised, vector, allow_pickle=False)
        await self.send(struct.pack("!BQQ", 0x03, id, serialised.tell()) + serialised.getbuffer())

    async def tclip_complete(self, id: int, vector: np.ndarray):
        print("tclip complete! sending vector")
        if self.dead: return

        serialised = BytesIO()
        np.save(serialised, vector, allow_pickle=False)
        await self.send(struct.pack("!BQQ", 0x04, id, serialised.tell()) + serialised.getbuffer())

async def main():
    while True:
        try:
            await Connection().run()
        except Exception as e:
            print(f"error with connection:")
            traceback.print_exc()
            await asyncio.sleep(10)

asyncio.run(main())