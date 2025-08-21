import os
from typing import List

import imageio
import numpy as np
import cv2


def _to_rgb_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    rgb_frames: List[np.ndarray] = []
    for f in frames:
        arr = np.asarray(f)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            b, g, r, a = np.split(arr, 4, axis=2)
            arr = np.concatenate([r, g, b], axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        else:
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
    # Try imageio ffmpeg writer with libx264, then default
    try:
        imageio.mimsave(path, rgb_frames, fps=fps, codec="libx264", quality=8, macro_block_size=None)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
    except Exception:
        pass
    try:
        imageio.mimsave(path, rgb_frames, fps=fps)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
    except Exception:
        pass
    # Fallback to OpenCV VideoWriter (mp4v)
    h, w = rgb_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not vw.isOpened():
        # last resort: write .avi
        alt_path = os.path.splitext(path)[0] + ".avi"
        fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(alt_path, fourcc2, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("无法创建视频文件（OpenCV VideoWriter 失败）")
        for fr in rgb_frames:
            bgr = fr[:, :, ::-1]
            vw.write(bgr)
        vw.release()
        return
    for fr in rgb_frames:
        bgr = fr[:, :, ::-1]
        vw.write(bgr)
    vw.release()
