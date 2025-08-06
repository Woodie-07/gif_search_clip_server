import numpy as np
import torch
import torch.nn as nn

from .utils.text_encoder import text_encoder
from .utils.vision_encoder import get_vision_encoder

from models.base import BaseModel

from threading import Lock

def normalize(data):
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    return (data / 255.0 - v_mean) / v_std

class VideoCLIP_XL(nn.Module):
    def __init__(self):
        super(VideoCLIP_XL, self).__init__()
        self.text_model = text_encoder.load().float()
        self.vision_model = get_vision_encoder().float()

class VideoCLIP_XL_v2(BaseModel):
    def __init__(self):
        self.lock = Lock()
        self.videoclip_xl = None

    def load(self) -> nn.Module:
        with self.lock:
            if self.videoclip_xl is None:
                print("loading VideoCLIP_XL_v2")
                self.videoclip_xl = VideoCLIP_XL()
                state_dict = torch.load("models/videoclip_xl_v2/VideoCLIP-XL-v2.bin", map_location="cpu")
                self.videoclip_xl.load_state_dict(state_dict)
                self.videoclip_xl.eval()
            return self.videoclip_xl

    def unload(self):
        with self.lock:
            if self.videoclip_xl is not None:
                print("unloading VideoCLIP_XL_v2")
                self.videoclip_xl = None

    def process_videos(self, videos: list[list[np.ndarray]]) -> list[np.ndarray]:
        print("VideoCLIP_XL_v2 processing videos")
        processed_videos = []
        for vid in videos:
            vid_tube = []
            for fr in vid:
                fr = fr[:, :, ::-1]  # BGR to RGB
                fr = np.expand_dims(normalize(fr), axis=(0, 1))  # (1,1,224,224,3)
                vid_tube.append(fr)

            vid_tube = np.concatenate(vid_tube, axis=1)  # (1, fnum, 224, 224, 3)
            vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # (1, fnum, 3, 224, 224)
            vid_tube = torch.from_numpy(vid_tube)
            processed_videos.append(vid_tube)
        processed_videos = torch.cat(processed_videos, 0).float()

        videoclip_xl = self.load()
        print("running model (v)")
        with torch.no_grad():
            video_features = videoclip_xl.vision_model.get_vid_features(processed_videos).float()
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features.numpy().tolist()

    def process_texts(self, texts: list[str]) -> list[np.ndarray]:
        videoclip_xl = self.load()
        print("running model (t)")
        with torch.no_grad():
            text_inputs = text_encoder.tokenize(texts, truncate=True)
            text_features = videoclip_xl.text_model.encode_text(text_inputs).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.numpy()
