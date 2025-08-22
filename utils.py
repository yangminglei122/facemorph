import os
import datetime as dt
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags
import re


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _is_wsl() -> bool:
    try:
        if os.name != "posix":
            return False
        with open("/proc/version", "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read().lower()
            return "microsoft" in txt or "wsl" in txt
    except Exception:
        return False


def normalize_path_for_runtime(path: str) -> str:
    """Expand ~ and, if running on WSL, convert Windows-style paths like C:\\Users\\... or C:/Users/... to /mnt/c/Users/...."""
    if not path:
        return path
    p = os.path.expanduser(path)
    if os.name == "posix" and p.startswith("/mnt/"):
        return p
    if os.name == "posix" and _is_wsl() and re.match(r"^[A-Za-z]:[\\/]", p):
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        mnt_root = f"/mnt/{drive}"
        if os.path.isdir(mnt_root):
            return f"{mnt_root}{rest}"
    return p


def get_file_mtime_dt(path: str) -> dt.datetime:
    return dt.datetime.fromtimestamp(os.path.getmtime(path))


def parse_date_from_filename(name: str) -> Optional[dt.datetime]:
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    s = stem
    patterns = [
        r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>[0-2][0-9]|3[01])[_-]?(?P<H>[0-1][0-9]|2[0-3])?(?P<M>[0-5][0-9])?(?P<S>[0-5][0-9])?",
        r"(?P<y>20\d{2})[-_.](?P<m>0[1-9]|1[0-2])[-_.](?P<d>[0-2][0-9]|3[01])(?:[ T](?P<H>[0-1][0-3]|2[0-3])[:_-]?(?P<M>[0-5][0-9])[:_-]?(?P<S>[0-5][0-9])?)?",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            y = int(m.group("y"))
            mo = int(m.group("m"))
            d = int(m.group("d"))
            H = int(m.group("H")) if m.group("H") else 0
            M = int(m.group("M")) if m.group("M") else 0
            S = int(m.group("S")) if m.group("S") else 0
            try:
                return dt.datetime(y, mo, d, H, M, S)
            except Exception:
                continue
    return None


def extract_first_integer(name: str) -> Optional[int]:
    m = re.search(r"(\d+)", os.path.basename(name))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def list_images_sorted(input_dir: str, sort: str = "name") -> List[str]:
    orig = input_dir
    input_dir = normalize_path_for_runtime(input_dir)
    # Friendly error if Windows path given on non-WSL POSIX
    if os.name == "posix" and not _is_wsl() and re.match(r"^[A-Za-z]:[\\/]", orig):
        raise FileNotFoundError(
            f"无法访问的路径: '{orig}'. 当前运行在 Linux (非 WSL)，无法直接读取 Windows 路径; 请将图片复制到本机路径(如 /home/...) 或在 WSL 中使用 /mnt/c/..."
        )
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"路径不存在: '{input_dir}'")
    files = []
    for name in os.listdir(input_dir):
        path = os.path.join(input_dir, name)
        if os.path.isfile(path) and os.path.splitext(name.lower())[1] in SUPPORTED_EXTS:
            files.append(path)
    if sort == "name":
        files.sort(key=lambda p: os.path.basename(p).lower())
    elif sort == "time":
        files.sort(key=lambda p: (read_exif_datetime(p) or get_file_mtime_dt(p)))
    elif sort == "exif":
        files.sort(key=lambda p: (read_exif_datetime(p) or get_file_mtime_dt(p)))
    elif sort == "filename_date":
        files.sort(key=lambda p: (parse_date_from_filename(p) or dt.datetime.min, os.path.basename(p).lower()))
    elif sort == "name_numeric":
        def _num_key(p: str):
            x = extract_first_integer(p)
            return (x if x is not None else float('inf'), os.path.basename(p).lower())
        files.sort(key=_num_key)
    else:
        raise ValueError(f"Unknown sort option: {sort}")
    return files


def read_exif_datetime(path: str) -> Optional[dt.datetime]:
    try:
        with Image.open(path) as img:
            exif = img._getexif() or {}
            if not exif:
                return None
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
                    try:
                        return dt.datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        continue
    except Exception:
        return None
    return None


@dataclass
class DetectionResult:
    face_points: Optional[np.ndarray]
    person_center: Optional[Tuple[float, float]]
    image_size: Tuple[int, int]


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def resize_with_padding(image: np.ndarray, target_w: int, target_h: int, border_mode=cv2.BORDER_REPLICATE) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    re = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    out = cv2.copyMakeBorder(re, pad_top, pad_bottom, pad_left, pad_right, border_mode)
    return out, scale, (pad_left, pad_top)


def ensure_dir(path: str) -> None:
    os.makedirs(normalize_path_for_runtime(path), exist_ok=True)


def compute_interocular_distance(face_points: np.ndarray) -> Optional[float]:
    if face_points is None or len(face_points) < 264:
        return None
    p_right = face_points[33]
    p_left = face_points[263]
    return float(np.linalg.norm(p_left - p_right))


def compute_roll_degrees(face_points: np.ndarray) -> float:
    """计算人脸旋转角度，基于眼睛关键点"""
    if face_points is None or len(face_points) < 468:
        return 0.0
    
    # 只使用最基本的眼睛关键点：33(右眼) 和 263(左眼)
    # 这是MediaPipe FaceMesh的标准索引
    right_eye_idx = 33
    left_eye_idx = 263
    
    if right_eye_idx >= len(face_points) or left_eye_idx >= len(face_points):
        return 0.0
    
    right_eye = face_points[right_eye_idx]
    left_eye = face_points[left_eye_idx]
    
    # 计算眼睛连线与水平线的夹角
    dy = float(left_eye[1] - right_eye[1])  # 左眼y - 右眼y
    dx = float(left_eye[0] - right_eye[0])  # 左眼x - 右眼x
    
    # 如果dx太小，说明眼睛几乎垂直，无法计算角度
    if abs(dx) < 1e-6:
        return 0.0
    
    # 计算角度
    angle_rad = np.arctan2(dy, dx)
    angle_deg = float(np.degrees(angle_rad))
    
    # 角度合理性检查
    if abs(angle_deg) > 45.0:
        print(f"警告：计算出的旋转角度过大 ({angle_deg:.2f}°)，可能检测有误，使用0°")
        return 0.0
    
    return angle_deg


def apply_affine_to_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    pts1 = np.hstack([points.astype(np.float32), ones])
    return (pts1 @ M.T).astype(np.float32)


def center_soft(image: np.ndarray, target_w: int, target_h: int, center_xy: Tuple[float, float]) -> Tuple[np.ndarray, float, Tuple[int, int], np.ndarray]:
    resized, scale, (pad_left, pad_top) = resize_with_padding(image, target_w, target_h)
    dx = int(round(target_w / 2 - (center_xy[0] * scale + pad_left)))
    dy = int(round(target_h / 2 - (center_xy[1] * scale + pad_top)))
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(resized, M, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return translated, scale, (pad_left, pad_top), M


def make_canvas_boundary_points(width: int, height: int) -> np.ndarray:
    w = width
    h = height
    pts = [
        (0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1),
        (w // 2, 0), (w - 1, h // 2), (w // 2, h - 1), (0, h // 2),
        (w // 4, 0), (3 * w // 4, 0), (w - 1, h // 4), (w - 1, 3 * h // 4),
        (3 * w // 4, h - 1), (w // 4, h - 1), (0, 3 * h // 4), (0, h // 4),
    ]
    return np.array(pts, dtype=np.float32)
