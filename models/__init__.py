from resize import ResizeMode
from models.base import BaseModel
from models.videoclip_xl_v2 import VideoCLIP_XL_v2
from models.xclip import XClip
from models.frozen_in_time import FrozenInTime

class Model:
    def __init__(self, name: str, model: BaseModel, dim: int, res: tuple[int, int], fcount: int, resize_mode: ResizeMode):
        self.name = name
        self.model = model
        self.dim = dim
        self.res = res
        self.fcount = fcount
        self.resize_mode = resize_mode

MODELS = [
    Model("VideoCLIP-XL-v2", VideoCLIP_XL_v2(), 768, (224, 224), 8, ResizeMode.STRETCH),
    Model("X-CLIP", XClip(), 768, (224, 224), 8, ResizeMode.NONE),
    Model("Frozen-in-Time", FrozenInTime(), 256, (224, 224), 4, ResizeMode.CENTER_CROP),
]
