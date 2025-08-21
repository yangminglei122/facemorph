from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 修复导入路径问题，确保在PyInstaller打包环境中能正确导入
try:
    # 尝试相对导入（开发环境）
    from utils import compute_interocular_distance, compute_roll_degrees, apply_affine_to_points
except ImportError:
    # 在PyInstaller打包环境中，尝试从项目根目录导入
    import sys
    import os
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 添加项目根目录到sys.path
        root_dir = sys._MEIPASS
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
    from utils import compute_interocular_distance, compute_roll_degrees, apply_affine_to_points


EYER_OUTER = 33
EYEL_OUTER = 263
MOUTH_R = 61
MOUTH_L = 291
NOSE_TIP = 1
CHIN = 199

# For frontal selection scoring
STABLE_INDEXES = [
    EYER_OUTER, EYEL_OUTER, MOUTH_R, MOUTH_L, NOSE_TIP, CHIN,
    70, 105, 334, 300, 168, 197, 5, 98, 327, 2
]

# Key anchors for precise similarity alignment
KEY_ALIGNMENT_INDEXES = [EYER_OUTER, EYEL_OUTER, MOUTH_R, MOUTH_L, NOSE_TIP]


@dataclass
class ReferenceGeometry:
    points: np.ndarray
    center_xy: Tuple[float, float]
    image_size: Tuple[int, int]
    triangles: List[Tuple[int, int, int]]


def select_reference_index(face_points_list: List[Optional[np.ndarray]]) -> int:
    scores = []
    for idx, pts in enumerate(face_points_list):
        if pts is None or len(pts) < 300:
            scores.append((1e9, idx))
            continue
        pr = pts[EYER_OUTER]
        pl = pts[EYEL_OUTER]
        mr = pts[MOUTH_R]
        ml = pts[MOUTH_L]
        den = (pl[0] - pr[0]) if abs(pl[0] - pr[0]) > 1e-6 else 1e-6
        slope = (pl[1] - pr[1]) / den
        mouth_sym = abs(np.linalg.norm(ml - pts.mean(axis=0)) - np.linalg.norm(mr - pts.mean(axis=0)))
        iod = compute_interocular_distance(pts) or 1.0
        score = abs(slope) * 1000.0 + mouth_sym * 0.1 - iod * 0.01
        scores.append((score, idx))
    scores.sort()
    return scores[0][1]


def _make_alignment_points(points: np.ndarray) -> np.ndarray:
    """Build enriched anchor points for robust similarity: eyes, nose tip, mouth corners, eye center, mouth center."""
    anchors = []
    for idx in KEY_ALIGNMENT_INDEXES:
        if idx < len(points):
            anchors.append(points[idx])
    # Eye center and mouth center
    if EYER_OUTER < len(points) and EYEL_OUTER < len(points):
        anchors.append((points[EYER_OUTER] + points[EYEL_OUTER]) * 0.5)
    if MOUTH_R < len(points) and MOUTH_L < len(points):
        anchors.append((points[MOUTH_R] + points[MOUTH_L]) * 0.5)
    return np.array(anchors, dtype=np.float32)


def estimate_similarity_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    if src_pts.shape[0] < 3 or dst_pts.shape[0] < 3:
        return None
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=3000, confidence=0.999)
    if M is None:
        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    return M


def _similarity_from_three(src_pts: np.ndarray, dst_pts: np.ndarray) -> Optional[np.ndarray]:
    if src_pts.shape[0] < 3 or dst_pts.shape[0] < 3:
        return None
    A = src_pts[:3].astype(np.float32)
    B = dst_pts[:3].astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(A, B, method=cv2.LMEDS)
    return M


def build_similarity_reference(face_pts_ref: np.ndarray, canvas_size: Tuple[int, int], subject_scale: float = 0.95, face_width_ratio: float = 0.33) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate reference to level eyes, center face, and set canonical face scale (interocular distance = face_width_ratio * width),
    then apply uniform subject_scale around canvas center to preserve background.
    Returns (ref_pts_target, M_ref_to_canvas).
    """
    h, w = canvas_size
    center = face_pts_ref.mean(axis=0).astype(np.float32)
    angle = compute_roll_degrees(face_pts_ref)
    M_rot = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle=-angle, scale=1.0)
    ref_pts_rot = apply_affine_to_points(face_pts_ref, M_rot)

    # Center
    ref_center = ref_pts_rot.mean(axis=0)
    dx = w / 2.0 - ref_center[0]
    dy = h / 2.0 - ref_center[1]
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    ref_pts_canvas = apply_affine_to_points(ref_pts_rot, M_trans)

    # Canonical interocular distance
    iod = compute_interocular_distance(ref_pts_canvas) or 1.0
    target_iod = max(1.0, w * face_width_ratio)
    s_face = float(target_iod / max(1e-6, iod))

    # Uniform scaling about canvas center to set face scale and preserve background
    cx, cy = w / 2.0, h / 2.0
    s_total = s_face * subject_scale
    S = np.float32([[s_total, 0, (1 - s_total) * cx], [0, s_total, (1 - s_total) * cy]])
    ref_pts_canvas = apply_affine_to_points(ref_pts_canvas, S)

    # Compose transforms: S @ T @ R
    M_ref = np.vstack([M_rot, [0, 0, 1]]).astype(np.float32)
    M_tr = np.vstack([M_trans, [0, 0, 1]]).astype(np.float32)
    M_sc = np.vstack([S, [0, 0, 1]]).astype(np.float32)
    M_total = (M_sc @ M_tr @ M_ref)[:2, :]
    return ref_pts_canvas, M_total


def warp_with_similarity(src_img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, canvas_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = canvas_size
    src_key = _make_alignment_points(src_pts)
    dst_key = _make_alignment_points(dst_pts)
    if src_key.shape[0] >= 3 and dst_key.shape[0] >= 3:
        M = estimate_similarity_transform(src_key, dst_key)
    else:
        M = None
    if M is None:
        M = _similarity_from_three(src_key, dst_key)
    if M is None:
        # Fallback: align centers only
        src_center = src_key.mean(axis=0) if len(src_key) else src_pts.mean(axis=0)
        dst_center = dst_key.mean(axis=0) if len(dst_key) else dst_pts.mean(axis=0)
        dx = dst_center[0] - src_center[0]
        dy = dst_center[1] - src_center[1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
    warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_pts = apply_affine_to_points(src_pts, M)
    # Small post-translation to perfect center match
    dst_center = dst_key.mean(axis=0) if len(dst_key) else dst_pts.mean(axis=0)
    warped_center = (_make_alignment_points(warped_pts).mean(axis=0) if len(_make_alignment_points(warped_pts)) else warped_pts.mean(axis=0))
    cx_err = float(dst_center[0] - warped_center[0])
    cy_err = float(dst_center[1] - warped_center[1])
    if abs(cx_err) > 2.0 or abs(cy_err) > 2.0:
        M_corr = np.float32([[1, 0, cx_err], [0, 1, cy_err]])
        warped = cv2.warpAffine(warped, M_corr, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_pts = apply_affine_to_points(warped_pts, M_corr)
    return warped, warped_pts
