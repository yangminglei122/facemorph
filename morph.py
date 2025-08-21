from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np


def _map_alpha(s: float, mode: str = "linear", a: float = 0.4, b: float = 0.6, p1: float = 2.2, p3: float = 0.8) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    a = float(np.clip(a, 0.0, 1.0))
    b = float(np.clip(b, 0.0, 1.0))
    if mode == "linear" or a >= b:
        return s
    if s <= a:
        return (s / a) ** p1 * a
    elif s >= b:
        return 1.0 - ((1.0 - s) / (1.0 - b)) ** p3 * (1.0 - b)
    else:
        return a + (s - a) * (b - a) / max(1e-6, (b - a))


def generate_crossfade_frames(
    images: List[np.ndarray],
    fps: int,
    transition_seconds: float,
    hold_seconds: float,
    easing: str = "linear",
    ease_a: float = 0.4,
    ease_b: float = 0.6,
    ease_p1: float = 2.2,
    ease_p3: float = 0.8,
) -> Generator[np.ndarray, None, None]:
    if not images:
        return
    T = max(1, int(round(transition_seconds * fps)))
    H = max(0, int(round(hold_seconds * fps)))
    for i in range(len(images) - 1):
        curr = images[i]
        next_img = images[i + 1]
        for _ in range(H):
            yield curr
        for t in range(1, T + 1):
            s = t / float(T)
            alpha = _map_alpha(s, mode=easing, a=ease_a, b=ease_b, p1=ease_p1, p3=ease_p3)
            frame = (curr.astype(np.float32) * (1 - alpha) + next_img.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
            yield frame
    for _ in range(H):
        yield images[-1]


def _points_to_face_mask(points: Optional[np.ndarray], size: Tuple[int, int], blur: int = 21) -> np.ndarray:
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    if points is None or len(points) == 0:
        return mask
    hull = cv2.convexHull(points.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, hull, 255)
    if blur > 0:
        k = blur if blur % 2 == 1 else blur + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    return mask


def _unsharp_mask(image: np.ndarray, amount: float = 0.3, radius: int = 3) -> np.ndarray:
    if amount <= 0:
        return image
    r = radius if radius % 2 == 1 else radius + 1
    blurred = cv2.GaussianBlur(image, (r, r), 0)
    sharp = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharp


def _has_cuda() -> bool:
    try:
        return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _farneback_flow(A_gray: np.ndarray, B_gray: np.ndarray, use_gpu: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Return forward and backward flow (H,W,2) float32. If use_gpu and CUDA available, compute with CUDA Farneback."""
    if use_gpu and _has_cuda():
        # Upload to GPU
        gA = cv2.cuda_GpuMat()
        gB = cv2.cuda_GpuMat()
        gA.upload(A_gray)
        gB.upload(B_gray)
        # Create CUDA Farneback
        fb = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=3,
            pyrScale=0.5,
            fastPyramids=False,
            winSize=35,
            numIters=3,
            polyN=7,
            polySigma=1.5,
            flags=0,
        )
        flow_f_gpu = fb.calc(gA, gB, None)
        flow_b_gpu = fb.calc(gB, gA, None)
        flow_f = flow_f_gpu.download()
        flow_b = flow_b_gpu.download()
        return flow_f.astype(np.float32), flow_b.astype(np.float32)
    # CPU fallback
    flow_f = cv2.calcOpticalFlowFarneback(A_gray, B_gray, None, pyr_scale=0.5, levels=3, winsize=35, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
    flow_b = cv2.calcOpticalFlowFarneback(B_gray, A_gray, None, pyr_scale=0.5, levels=3, winsize=35, iterations=3, poly_n=7, poly_sigma=1.5, flags=0)
    return flow_f.astype(np.float32), flow_b.astype(np.float32)


def generate_flow_morph_frames(
    images: List[np.ndarray],
    aligned_points: List[Optional[np.ndarray]],
    fps: int,
    transition_seconds: float,
    hold_seconds: float,
    flow_strength: float = 1.0,
    face_protect: float = 0.6,
    sharpen_amount: float = 0.2,
    easing: str = "linear",
    ease_a: float = 0.4,
    ease_b: float = 0.6,
    ease_p1: float = 2.2,
    ease_p3: float = 0.8,
    use_gpu: bool = False,
) -> Generator[np.ndarray, None, None]:
    if not images:
        return
    T = max(1, int(round(transition_seconds * fps)))
    H = max(0, int(round(hold_seconds * fps)))
    h, w = images[0].shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    for i in range(len(images) - 1):
        A = images[i]
        B = images[i + 1]
        for _ in range(H):
            yield A
        A_gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
        B_gray = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)
        flow_f, flow_b = _farneback_flow(A_gray, B_gray, use_gpu=use_gpu)
        maskA = _points_to_face_mask(aligned_points[i] if i < len(aligned_points) else None, (h, w), blur=31)
        maskB = _points_to_face_mask(aligned_points[i + 1] if i + 1 < len(aligned_points) else None, (h, w), blur=31)
        maskA = (maskA.astype(np.float32) / 255.0)
        maskB = (maskB.astype(np.float32) / 255.0)
        protectA = 1.0 - face_protect * maskA
        protectB = 1.0 - face_protect * maskB
        for t in range(1, T + 1):
            s = t / float(T)
            alpha = _map_alpha(s, mode=easing, a=ease_a, b=ease_b, p1=ease_p1, p3=ease_p3)
            fxA = grid_x + (flow_f[..., 0] * (alpha * flow_strength) * protectA)
            fyA = grid_y + (flow_f[..., 1] * (alpha * flow_strength) * protectA)
            fxB = grid_x + (flow_b[..., 0] * ((1.0 - alpha) * flow_strength) * protectB)
            fyB = grid_y + (flow_b[..., 1] * ((1.0 - alpha) * flow_strength) * protectB)
            warpA = cv2.remap(A, fxA, fyA, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            warpB = cv2.remap(B, fxB, fyB, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            frame = (warpA.astype(np.float32) * (1 - alpha) + warpB.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
            if sharpen_amount > 0:
                frame = _unsharp_mask(frame, amount=sharpen_amount, radius=3)
            yield frame
    for _ in range(H):
        yield images[-1]
