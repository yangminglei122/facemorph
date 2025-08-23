import os
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import argparse

import cv2
import numpy as np
from PIL import Image, ExifTags
import re
import gc

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
        r"(?P<y>19\d{2}|20\d{2})(?P<m>0[1-9]|1[0-2])(?P<d>[0-2][0-9]|3[01])[_-]?(?P<H>[0-1][0-9]|2[0-3])?(?P<M>[0-5][0-9])?(?P<S>[0-5][0-9])?",
        r"(?P<y>19\d{2}|20\d{2})[-_.](?P<m>0[1-9]|1[0-2])[-_.](?P<d>[0-2][0-9]|3[01])(?:[ T](?P<H>[0-1][0-3]|2[0-3])[:_-]?(?P<M>[0-5][0-9])[:_-]?(?P<S>[0-5][0-9])?)?",
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
    """计算人脸旋转角度，基于眼睛、鼻子、嘴巴、下颌等关键面部特征点"""
    if face_points is None or len(face_points) < 468:
        return 0.0
    
    # 定义关键面部特征点索引（MediaPipe FaceMesh）
    # 眼睛关键点
    eye_points = {
        'right_outer': 33,   # 右眼外角
        'right_inner': 133,  # 右眼内角
        'left_inner': 362,   # 左眼内角
        'left_outer': 263,   # 左眼外角
    }
    
    # 鼻子关键点
    nose_points = {
        'tip': 1,        # 鼻尖
        'bridge': 168,   # 鼻梁
        'left': 129,     # 左鼻翼
        'right': 358,    # 右鼻翼
    }
    
    # 嘴巴关键点
    mouth_points = {
        'left': 61,      # 左嘴角
        'right': 291,    # 右嘴角
        'top': 13,       # 上唇中心
        'bottom': 14,    # 下唇中心
    }
    
    # 下颌关键点
    jaw_points = {
        'chin': 199,     # 下巴中心
        'left': 132,     # 左下颌
        'right': 361,    # 右下颌
    }
    
    angles = []
    
    # 方法1：眼睛连线角度（最重要）
    if all(idx < len(face_points) for idx in eye_points.values()):
        # 使用外眼角连线
        right_outer = face_points[eye_points['right_outer']]
        left_outer = face_points[eye_points['left_outer']]
        dy = float(left_outer[1] - right_outer[1])
        dx = float(left_outer[0] - right_outer[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            normalized_angle = normalize_angle(angle_deg)
            angles.append(normalized_angle)
        
        # 使用内眼角连线
        right_inner = face_points[eye_points['right_inner']]
        left_inner = face_points[eye_points['left_inner']]
        dy = float(left_inner[1] - right_inner[1])
        dx = float(left_inner[0] - right_inner[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            normalized_angle = normalize_angle(angle_deg)
            angles.append(normalized_angle)
    
    # 方法2：鼻子水平线角度
    if all(idx < len(face_points) for idx in [nose_points['left'], nose_points['right']]):
        nose_left = face_points[nose_points['left']]
        nose_right = face_points[nose_points['right']]
        dy = float(nose_left[1] - nose_right[1])
        dx = float(nose_left[0] - nose_right[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            normalized_angle = normalize_angle(angle_deg)
            angles.append(normalized_angle)
    
    # 方法3：嘴巴连线角度
    if all(idx < len(face_points) for idx in [mouth_points['left'], mouth_points['right']]):
        mouth_left = face_points[mouth_points['left']]
        mouth_right = face_points[mouth_points['right']]
        dy = float(mouth_left[1] - mouth_right[1])
        dx = float(mouth_left[0] - mouth_right[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            normalized_angle = normalize_angle(angle_deg)
            angles.append(normalized_angle)
    
    # 方法4：下颌连线角度
    if all(idx < len(face_points) for idx in [jaw_points['left'], jaw_points['right']]):
        jaw_left = face_points[jaw_points['left']]
        jaw_right = face_points[jaw_points['right']]
        dy = float(jaw_left[1] - jaw_right[1])
        dx = float(jaw_left[0] - jaw_right[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            normalized_angle = normalize_angle(angle_deg)
            angles.append(normalized_angle)
    
    # 方法5：面部中心线角度（眼睛中心到下巴中心）
    if (all(idx < len(face_points) for idx in [eye_points['right_outer'], eye_points['left_outer']]) and
        jaw_points['chin'] < len(face_points)):
        # 计算眼睛中心
        right_eye = face_points[eye_points['right_outer']]
        left_eye = face_points[eye_points['left_outer']]
        eye_center = ((right_eye[0] + left_eye[0]) / 2.0, (right_eye[1] + left_eye[1]) / 2.0)
        
        # 下巴中心
        chin = face_points[jaw_points['chin']]
        
        # 计算面部中心线角度
        dy = float(chin[1] - eye_center[1])
        dx = float(chin[0] - eye_center[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            # 面部中心线应该是垂直的，所以角度应该是90度
            # 我们计算的是相对于水平线的角度，所以需要调整
            face_angle = float(np.degrees(angle_rad)) - 90.0
            normalized_face_angle = normalize_angle(face_angle)
            angles.append(normalized_face_angle)
    
    if not angles:
        # 回退到原来的方法
        if 33 < len(face_points) and 263 < len(face_points):
            right_eye = face_points[33]
            left_eye = face_points[263]
            dy = float(left_eye[1] - right_eye[1])
            dx = float(left_eye[0] - right_eye[0])
            if abs(dx) < 1e-6:
                return 0.0
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            normalized_angle = normalize_angle(angle_deg)
            return normalized_angle
        return 0.0
    
    # 使用加权中位数，眼睛权重最高
    if len(angles) >= 3:
        # 前两个角度是眼睛的，权重更高
        eye_angles = angles[:2] if len(angles) >= 2 else angles
        other_angles = angles[2:] if len(angles) >= 3 else []
        
        # 计算眼睛角度的中位数
        eye_median = float(np.median(eye_angles))
        
        # 如果有其他角度，进行加权平均
        if other_angles:
            other_median = float(np.median(other_angles))
            # 眼睛权重0.7，其他特征权重0.3
            final_angle = eye_median * 0.7 + other_median * 0.3
        else:
            final_angle = eye_median
    else:
        # 角度数量不足，使用简单中位数
        final_angle = float(np.median(angles))
    
    # 角度合理性检查
    if abs(final_angle) > 30.0:  # 降低阈值，更严格
        print(f"警告：计算出的旋转角度过大 ({final_angle:.2f}°)，可能检测有误，使用0°")
        return 0.0
    
    return final_angle


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


def normalize_angle(angle_deg: float) -> float:
    """标准化角度到 -90° 到 90° 范围
    
    Args:
        angle_deg: 输入角度（度）
    
    Returns:
        标准化后的角度（度）
    
    说明：
    - 对于水平线，0° 和 180° 表示相同方向，标准化为 0°
    - 对于垂直线，90° 和 -90° 表示相同方向，标准化为 90°
    - 角度范围标准化到 -90° 到 90°
    """
    # 将角度标准化到 -180° 到 180° 范围
    angle_deg = angle_deg % 360
    if angle_deg > 180:
        angle_deg -= 360
    elif angle_deg < -180:
        angle_deg += 360
    
    # 将角度标准化到 -90° 到 90° 范围
    if angle_deg > 90:
        angle_deg = 180 - angle_deg
    elif angle_deg < -90:
        angle_deg = -180 - angle_deg
    
    return angle_deg

def extract_timestamps(paths: List[str], sort: str) -> List[Optional[dt.datetime]]:
    """从图片路径中提取时间戳信息"""
    timestamps = []
    for path in paths:
        # 优先使用文件名时间
        timestamp = parse_date_from_filename(path)
        if timestamp is None:
            # 尝试使用EXIF时间
            timestamp = read_exif_datetime(path)
        if timestamp is None:
            # 使用文件修改时间
            timestamp = get_file_mtime_dt(path)
        timestamps.append(timestamp)
    return timestamps


def load_images(paths: List[str]) -> List[np.ndarray]:
    """加载多张图像到内存"""
    images = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取图像: {p}")
        images.append(img)
    return images


def get_memory_usage() -> float:
    """获取当前内存使用量（MB）"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # 转换为MB
    except ImportError:
        return 0.0


def process_batch(batch_paths: List[str], batch_timestamps: List[Optional[dt.datetime]],
                 ref_pts_target: np.ndarray, config: Union[Dict[str, Any], argparse.Namespace], start_idx: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[dt.datetime]]]:
    """处理单个批次的图像"""
    from align import warp_with_similarity
    from detect import LandmarkDetector
    
    # 加载当前批次图像
    batch_images = load_images(batch_paths)
    
    # 检测人脸关键点
    detector = LandmarkDetector()
    batch_detections = []
    for img in batch_images:
        batch_detections.append(detector.detect(img))
    detector.close()
    
    batch_face_points = [d.face_points for d in batch_detections]
    
    # 对齐图像
    aligned_images = []
    aligned_points = []
    aligned_timestamps = []
    
    # 处理config参数，支持字典和argparse.Namespace两种类型
    if hasattr(config, 'get'):
        # 字典类型
        width = int(config.get("width", 1080))
        height = int(config.get("height", 1350))
    else:
        # argparse.Namespace类型
        width = int(getattr(config, "width", 1080))
        height = int(getattr(config, "height", 1350))
    
    for i, (img, pts) in enumerate(zip(batch_images, batch_face_points)):
        if pts is not None and len(pts) >= 300:
            try:
                # 对齐到参考图
                result = warp_with_similarity(img, pts, ref_pts_target, (height, width))
                if result is not None:
                    aligned_img, aligned_pts = result
                    aligned_images.append(aligned_img)
                    aligned_points.append(aligned_pts)
                    aligned_timestamps.append(batch_timestamps[i])
            except Exception as e:
                # 忽略对齐失败的图像
                print(f"警告：图像 {i} 对齐失败: {e}")
                continue
        else:
            # 人脸检测或关键点检测失败
            filename = os.path.basename(batch_paths[i])
            global_index = start_idx + i  # 计算全局索引
            print(f"警告：图像 {global_index} 人脸检测或关键点检测失败，文件: {filename}")
            continue
    
    # 释放当前批次的内存
    del batch_images
    del batch_detections
    del batch_face_points
    gc.collect()
    
    return aligned_images, aligned_points, aligned_timestamps
