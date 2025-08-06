from abc import ABC
import numpy as np
import torch.nn as nn

class BaseModel(ABC):
    def __init__(self):
        pass

    def load(self) -> nn.Module:
        pass

    def unload(self):
        pass

    def process_videos(self, videos: list[list[np.ndarray]]) -> list[np.ndarray]:
        pass

    def process_texts(self, texts: list[str]) -> list[np.ndarray]:
        pass
