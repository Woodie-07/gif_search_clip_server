from transformers import XCLIPModel, XCLIPProcessor
from models.base import BaseModel
from threading import Lock
import numpy as np
import torch

def normalize(data):
    v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    return (data / 255.0 - v_mean) / v_std

class XClip(BaseModel):
    def __init__(self):
        self.lock = Lock()
        self.model = None
        self.processor = None

    def load(self) -> tuple[XCLIPModel, XCLIPProcessor]:
        with self.lock:
            if self.model is None:
                print("loading X-CLIP")
                self.model = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32")
                self.processor = XCLIPProcessor.from_pretrained("microsoft/xclip-base-patch32")
                self.model.eval()
            return self.model, self.processor

    def unload(self):
        with self.lock:
            if self.model is not None:
                print("unloading X-CLIP")
                self.model = None
                self.processor = None

    def process_videos(self, videos: list[list[np.ndarray]]) -> list[np.ndarray]:
        print(f"X-CLIP processing {len(videos)} videos")
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

        model, _ = self.load()
        print("running model (v)")
        with torch.no_grad():
            video_features = model.get_video_features(processed_videos).float()
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features.numpy().tolist()

    def process_texts(self, texts: list[str]) -> list[np.ndarray]:
        model, processor = self.load()
        print("running model (t)")
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.numpy()
