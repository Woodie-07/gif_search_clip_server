from transformers import XCLIPModel, XCLIPProcessor
from models.base import BaseModel
from threading import Lock
import numpy as np
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

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
                self.model = XCLIPModel.from_pretrained("microsoft/xclip-large-patch14-kinetics-600").to(device)
                self.processor = XCLIPProcessor.from_pretrained("microsoft/xclip-large-patch14-kinetics-600")
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

        model, processor = self.load()
        print("running X-CLIP (v)")
        with torch.no_grad():
            video_features = model.get_video_features(**processor(videos=videos, return_tensors="pt").to(device)).float()
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)

        return video_features.cpu().numpy().tolist()

    def process_texts(self, texts: list[str]) -> list[np.ndarray]:
        model, processor = self.load()
        print("running X-CLIP (t)")
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy()
