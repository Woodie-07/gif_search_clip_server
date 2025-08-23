from enum import IntEnum, auto
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

class ResizeMode(IntEnum):
    STRETCH = auto()
    CENTER_CROP = auto()
    NONE = auto()
    

def _resize_image(image: Image.Image, resolution: tuple[int, int], resize_mode: ResizeMode) -> Image.Image:
    if resize_mode == ResizeMode.STRETCH:
        return image.resize(resolution)
    elif resize_mode == ResizeMode.CENTER_CROP:
        new_width, new_height = image.size
        new_top = 0
        new_left = 0
        if new_width < new_height:
            new_top = (new_height - new_width) // 2
            new_height = new_width
        else:
            new_left = (new_width - new_height) // 2
            new_width = new_height
        return image.resize(resolution, box=(new_left, new_top, new_left + new_width, new_top + new_height))
    elif resize_mode == ResizeMode.NONE:
        return image
    else:
        raise ValueError("Invalid resize mode")

def _resize_mat_cv2(mat: np.ndarray, resolution: tuple[int, int], resize_mode: ResizeMode) -> np.ndarray:
    if resize_mode == ResizeMode.STRETCH:
        return cv2.resize(mat, resolution)
    elif resize_mode == ResizeMode.CENTER_CROP:
        new_height, new_width = mat.shape[:2]
        new_top = 0
        new_left = 0
        if new_width < new_height:
            new_top = (new_height - new_width) // 2
            new_height = new_width
        else:
            new_left = (new_width - new_height) // 2
            new_width = new_height
        return cv2.resize(mat[new_top:new_top + new_height, new_left:new_left + new_width], resolution)
    elif resize_mode == ResizeMode.NONE:
        return mat
    else:
        raise ValueError("Invalid resize mode")

def resize_media(data: BytesIO, content_type: str, nframes: int, resolution: tuple[int, int], resize_mode: ResizeMode) -> list[np.ndarray]:
    frames = []
    if content_type in ("image/gif", "image/png"):
        image = Image.open(data)
        if content_type == "image/gif":
            if image.n_frames == 0:
                return []
            if image.n_frames < nframes:
                for i in range(image.n_frames):
                    image.seek(i)
                    frames.append(np.array(_resize_image(image.convert("RGB"), resolution, resize_mode)))
                frames.extend([frames[-1]] * (nframes - len(frames)))
            else:
                step = image.n_frames / nframes
                for i in range(nframes):
                    image.seek(int(i * step))
                    frames.append(np.array(_resize_image(image.convert("RGB"), resolution, resize_mode)))
        else:
            frames = [np.array(_resize_image(image.convert("RGB"), resolution, resize_mode))] * nframes
    else:
        video = cv2.VideoCapture(source=data, apiPreference=cv2.CAP_FFMPEG, params=[])
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []
        if total_frames < nframes:
            for _ in range(total_frames):
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(_resize_mat_cv2(frame, resolution, resize_mode))
            frames.extend([frames[-1]] * (nframes - len(frames)))
        else:
            step = total_frames / nframes
            for i in range(nframes):
                video.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(_resize_mat_cv2(frame, resolution, resize_mode))
    return frames