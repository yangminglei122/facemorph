import os
from typing import List

import imageio
import numpy as np


def _to_rgb_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    rgb_frames: List[np.ndarray] = []
    for f in frames:
        arr = np.asarray(f)
        # Normalize dtype to uint8 in [0,255]
        if arr.dtype != np.uint8:
            # Assume values are 0..255; clip and cast
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Ensure 3 channels
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            # BGRA -> RGBA
            b, g, r, a = np.split(arr, 4, axis=2)
            arr = np.concatenate([r, g, b, a], axis=2)
            # Drop alpha for GIF/MP4 compatibility
            arr = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] == 3:
            # BGR -> RGB
            arr = arr[:, :, ::-1]
        else:
            # Fallback: reshape if somehow 1x1x3, keep as is after cast
            arr = arr.reshape((arr.shape[0], arr.shape[1], -1))
            if arr.shape[2] == 3:
                arr = arr[:, :, ::-1]
        rgb_frames.append(arr)
    return rgb_frames


def save_gif(frames: List[np.ndarray], path: str, fps: int = 15) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration = 1.0 / max(1, fps)
    rgb_frames = _to_rgb_frames(frames)
    imageio.mimsave(path, rgb_frames, format="GIF", duration=duration)


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rgb_frames = _to_rgb_frames(frames)
    imageio.mimsave(path, rgb_frames, fps=fps, codec="libx264", quality=8)
