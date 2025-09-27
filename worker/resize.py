from enum import IntEnum, auto
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

class ResizeMode(IntEnum):
    STRETCH = auto()
    CENTER_CROP = auto()
    NONE = auto()

# resize a pillow image to the given resolution
def _resize_image(image: Image.Image, resolution: tuple[int, int], resize_mode: ResizeMode) -> Image.Image:
    if resize_mode == ResizeMode.STRETCH: # stretch the image to fit the resolution
        return image.resize(resolution)
    elif resize_mode == ResizeMode.CENTER_CROP:
        # try to retain as much as the image as possible while cropping either from the top and bottom or from the sides
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
    elif resize_mode == ResizeMode.NONE: # leave image as-is. used when the resizing is done in the model-specific code
        return image
    else:
        raise ValueError("Invalid resize mode")

# resize an image from a numpy array
def _resize_mat_cv2(mat: np.ndarray, resolution: tuple[int, int], resize_mode: ResizeMode) -> np.ndarray:
    if resize_mode == ResizeMode.STRETCH: # stretch the image to fit the resolution
        return cv2.resize(mat, resolution)
    elif resize_mode == ResizeMode.CENTER_CROP:
        # try to retain as much as the image as possible while cropping either from the top and bottom or from the sides
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
    elif resize_mode == ResizeMode.NONE: # leave image as-is. used when the resizing is done in the model-specific code
        return mat
    else:
        raise ValueError("Invalid resize mode")

def resize_media(data: BytesIO, content_type: str, nframes: int, resolution: tuple[int, int], resize_mode: ResizeMode) -> list[np.ndarray]:
    frames = []
    if content_type.startswith("image/"):
        image = Image.open(data)
        if content_type == "image/gif":
            if image.n_frames == 0:
                return []
            if image.n_frames < nframes:
                # clone the final frame to fill the remaining frames
                for i in range(image.n_frames):
                    image.seek(i)
                    frames.append(np.array(_resize_image(image.convert("RGB"), resolution, resize_mode)))
                frames.extend([frames[-1]] * (nframes - len(frames)))
            else:
                # take frames evenly spread throughout
                step = image.n_frames / nframes
                for i in range(nframes):
                    image.seek(int(i * step))
                    frames.append(np.array(_resize_image(image.convert("RGB"), resolution, resize_mode)))
        else:
            # static image, simply copy it nframes times
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
            # read the video in sequence while picking out frames that give nframes evenly spread
            step = total_frames / nframes
            desired_frames = {int(i * step) for i in range(nframes)}
            for i in range(total_frames):
                ret, frame = video.read()
                if not ret:
                    raise ValueError("Failed to read frame from video")
                if i in desired_frames:
                    frames.append(_resize_mat_cv2(frame, resolution, resize_mode))
                    if len(frames) == nframes:
                        break
    return frames
