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


def select_reference_index(face_points_list: List[Optional[np.ndarray]], manual_ref_idx: Optional[int] = None) -> int:
    """选择参考图索引
    Args:
        face_points_list: 人脸关键点列表
        manual_ref_idx: 手动指定的参考图索引，如果为None则自动选择
    Returns:
        参考图索引
    """
    if manual_ref_idx is not None:
        if 0 <= manual_ref_idx < len(face_points_list):
            pts = face_points_list[manual_ref_idx]
            if pts is not None and len(pts) >= 300:
                # 验证手动指定的参考图质量
                quality_score = _compute_face_quality_score(pts)
                print(f"手动指定参考图索引 {manual_ref_idx}，质量评分: {quality_score:.2f}")
                if quality_score < 0.5:  # 质量太差时给出警告
                    print(f"警告：手动指定的参考图质量较低，建议选择更正面的图像")
                return manual_ref_idx
            else:
                print(f"警告：手动指定的参考图索引 {manual_ref_idx} 未检测到有效人脸关键点，将自动选择")
        else:
            print(f"警告：手动指定的参考图索引 {manual_ref_idx} 超出范围，将自动选择")
    
    # 自动选择逻辑
    scores = []
    for idx, pts in enumerate(face_points_list):
        if pts is None or len(pts) < 300:
            scores.append((1e9, idx))
            continue
        quality_score = _compute_face_quality_score(pts)
        scores.append((1.0 - quality_score, idx))  # 质量越高，分数越低
    
    scores.sort()
    best_idx = scores[0][1]
    best_score = 1.0 - scores[0][0]
    print(f"自动选择参考图索引 {best_idx}，质量评分: {best_score:.2f}")
    return best_idx


def _compute_face_quality_score(pts: np.ndarray) -> float:
    """计算人脸质量评分（0-1，越高越好），基于眼睛、鼻子、嘴巴、下颌等关键特征"""
    if pts is None or len(pts) < 468:
        return 0.0
    
    # 定义关键面部特征点索引
    eye_points = {
        'right_outer': 33,   # 右眼外角
        'right_inner': 133,  # 右眼内角
        'left_inner': 362,   # 左眼内角
        'left_outer': 263,   # 左眼外角
    }
    
    nose_points = {
        'tip': 1,        # 鼻尖
        'bridge': 168,   # 鼻梁
        'left': 129,     # 左鼻翼
        'right': 358,    # 右鼻翼
    }
    
    mouth_points = {
        'left': 61,      # 左嘴角
        'right': 291,    # 右嘴角
        'top': 13,       # 上唇中心
        'bottom': 14,    # 下唇中心
    }
    
    jaw_points = {
        'chin': 199,     # 下巴中心
        'left': 132,     # 左下颌
        'right': 361,    # 右下颌
    }
    
    scores = []
    
    # 1. 眼部水平度（最重要，权重0.4）
    if all(idx < len(pts) for idx in eye_points.values()):
        # 使用外眼角连线
        right_outer = pts[eye_points['right_outer']]
        left_outer = pts[eye_points['left_outer']]
        den = (left_outer[0] - right_outer[0]) if abs(left_outer[0] - right_outer[0]) > 1e-6 else 1e-6
        slope = abs((left_outer[1] - right_outer[1]) / den)
        eye_level_score = max(0.0, 1.0 - slope * 8.0)  # 调整系数
        scores.append(('eye_level', eye_level_score, 0.4))
        
        # 使用内眼角连线作为验证
        right_inner = pts[eye_points['right_inner']]
        left_inner = pts[eye_points['left_inner']]
        den = (left_inner[0] - right_inner[0]) if abs(left_inner[0] - right_inner[0]) > 1e-6 else 1e-6
        slope = abs((left_inner[1] - right_inner[1]) / den)
        eye_inner_score = max(0.0, 1.0 - slope * 8.0)
        scores.append(('eye_inner', eye_inner_score, 0.1))
    
    # 2. 鼻子对称性（权重0.2）
    if all(idx < len(pts) for idx in [nose_points['left'], nose_points['right'], nose_points['tip']]):
        nose_left = pts[nose_points['left']]
        nose_right = pts[nose_points['right']]
        nose_tip = pts[nose_points['tip']]
        
        # 检查鼻子是否在面部中心
        nose_center = (nose_left + nose_right) / 2.0
        face_center = pts.mean(axis=0)
        nose_offset = abs(nose_center[0] - face_center[0]) / np.linalg.norm(nose_left - nose_right)
        nose_score = max(0.0, 1.0 - nose_offset * 3.0)
        scores.append(('nose_sym', nose_score, 0.2))
    
    # 3. 嘴巴对称性（权重0.15）
    if all(idx < len(pts) for idx in [mouth_points['left'], mouth_points['right']]):
        mouth_left = pts[mouth_points['left']]
        mouth_right = pts[mouth_points['right']]
        
        # 检查嘴巴对称性
        mouth_center = (mouth_left + mouth_right) / 2.0
        face_center = pts.mean(axis=0)
        mouth_offset = abs(mouth_center[0] - face_center[0]) / np.linalg.norm(mouth_left - mouth_right)
        mouth_score = max(0.0, 1.0 - mouth_offset * 3.0)
        scores.append(('mouth_sym', mouth_score, 0.15))
    
    # 4. 下颌对称性（权重0.1）
    if all(idx < len(pts) for idx in [jaw_points['left'], jaw_points['right'], jaw_points['chin']]):
        jaw_left = pts[jaw_points['left']]
        jaw_right = pts[jaw_points['right']]
        jaw_center = (jaw_left + jaw_right) / 2.0
        chin = pts[jaw_points['chin']]
        
        # 检查下颌对称性
        face_center = pts.mean(axis=0)
        jaw_offset = abs(jaw_center[0] - face_center[0]) / np.linalg.norm(jaw_left - jaw_right)
        jaw_score = max(0.0, 1.0 - jaw_offset * 3.0)
        scores.append(('jaw_sym', jaw_score, 0.1))
    
    # 5. 面部整体对称性（权重0.05）
    if len(scores) >= 3:
        # 计算所有特征点的对称性
        symmetry_scores = [score for _, score, _ in scores]
        overall_sym_score = np.mean(symmetry_scores)
        scores.append(('overall_sym', overall_sym_score, 0.05))
    
    # 计算加权总分
    if not scores:
        return 0.0
    
    total_score = 0.0
    total_weight = 0.0
    
    for name, score, weight in scores:
        total_score += score * weight
        total_weight += weight
    
    # 归一化到0-1范围
    if total_weight > 0:
        final_score = total_score / total_weight
    else:
        final_score = 0.0
    
    return min(1.0, max(0.0, final_score))


def process_external_reference_image(ref_image_path: str, detector) -> Optional[np.ndarray]:
    """处理外部参考图像
    Args:
        ref_image_path: 外部参考图像路径
        detector: 人脸检测器
    Returns:
        检测到的人脸关键点，如果检测失败返回None
    """
    try:
        # 读取图像
        img = cv2.imdecode(np.fromfile(ref_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(ref_image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"无法读取外部参考图像: {ref_image_path}")
            return None
        
        # 检测人脸关键点
        detection = detector.detect(img)
        if detection.face_points is not None and len(detection.face_points) >= 300:
            return detection.face_points
        else:
            print(f"外部参考图像未检测到有效人脸关键点: {ref_image_path}")
            return None
    except Exception as e:
        print(f"处理外部参考图像时出错: {e}")
        return None


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
    
    # 改进的旋转角度计算
    angle = compute_roll_degrees(face_pts_ref)
    
    # 角度稳定性检查
    if abs(angle) < 0.5:  # 角度很小，不需要旋转
        print(f"参考图对齐：旋转角度 {angle:.2f}° (角度很小，跳过旋转)")
        ref_pts_rot = face_pts_ref.copy()
        M_rot = np.float32([[1, 0, 0], [0, 1, 0]])
    else:
        # 使用眼部中心作为旋转中心，而不是面部中心
        pr = face_pts_ref[EYER_OUTER]
        pl = face_pts_ref[EYEL_OUTER]
        eye_center = ((pr[0] + pl[0]) / 2.0, (pr[1] + pl[1]) / 2.0)
        
        # 应用旋转
        M_rot = cv2.getRotationMatrix2D(eye_center, angle=angle, scale=1.0)
    ref_pts_rot = apply_affine_to_points(face_pts_ref, M_rot)

    # 改进的居中算法：使用眼部中心对齐到画布中心
    pr_rot = ref_pts_rot[EYER_OUTER]
    pl_rot = ref_pts_rot[EYEL_OUTER]
    eye_center_rot = ((pr_rot[0] + pl_rot[0]) / 2.0, (pr_rot[1] + pl_rot[1]) / 2.0)
    
    dx = w / 2.0 - eye_center_rot[0]
    dy = h / 2.0 - eye_center_rot[1]
    M_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    ref_pts_canvas = apply_affine_to_points(ref_pts_rot, M_trans)

    # 改进的缩放算法：基于瞳距进行缩放
    iod = compute_interocular_distance(ref_pts_canvas) or 1.0
    target_iod = max(1.0, w * face_width_ratio)
    s_face = float(target_iod / max(1e-6, iod))

    # 以眼部中心为基准进行缩放
    eye_center_canvas = ((ref_pts_canvas[EYER_OUTER][0] + ref_pts_canvas[EYEL_OUTER][0]) / 2.0,
                        (ref_pts_canvas[EYER_OUTER][1] + ref_pts_canvas[EYEL_OUTER][1]) / 2.0)
    
    s_total = s_face * subject_scale
    S = np.float32([[s_total, 0, (1 - s_total) * eye_center_canvas[0]], 
                    [0, s_total, (1 - s_total) * eye_center_canvas[1]]])
    ref_pts_canvas = apply_affine_to_points(ref_pts_canvas, S)

    # 组合变换：S @ T @ R
    M_ref = np.vstack([M_rot, [0, 0, 1]]).astype(np.float32)
    M_tr = np.vstack([M_trans, [0, 0, 1]]).astype(np.float32)
    M_sc = np.vstack([S, [0, 0, 1]]).astype(np.float32)
    M_total = (M_sc @ M_tr @ M_ref)[:2, :]
    
    # 验证对齐效果
    final_angle = _verify_alignment_quality(ref_pts_canvas)
    
    print(f"参考图对齐：旋转角度 {angle:.2f}°，缩放比例 {s_total:.3f}，最终眼部角度 {final_angle:.2f}°")
    return ref_pts_canvas, M_total


def _verify_alignment_quality(aligned_points: np.ndarray) -> float:
    """验证对齐后的质量，返回眼部水平度"""
    if aligned_points is None or len(aligned_points) < 468:
        return 0.0
    
    # 检查对齐后的眼部水平度
    try:
        right_eye = aligned_points[33]
        left_eye = aligned_points[263]
        dy = float(left_eye[1] - right_eye[1])
        dx = float(left_eye[0] - right_eye[0])
        if abs(dx) < 1e-6:
            return 0.0
        angle_rad = np.arctan2(dy, dx)
        angle_deg = float(np.degrees(angle_rad))
        # 使用标准化角度，确保角度在合理范围内
        from utils import normalize_angle
        normalized_angle = normalize_angle(angle_deg)
        return normalized_angle
    except Exception:
        return 0.0


def warp_with_similarity(src_img: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray, canvas_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = canvas_size
    
    # 改进的对齐点选择：优先使用眼部关键点
    src_key = _make_improved_alignment_points(src_pts)
    dst_key = _make_improved_alignment_points(dst_pts)
    
    if src_key.shape[0] >= 3 and dst_key.shape[0] >= 3:
        M = estimate_similarity_transform(src_key, dst_key)
    else:
        M = None
    if M is None:
        M = _similarity_from_three(src_key, dst_key)
    if M is None:
        # Fallback: 基于眼部中心对齐
        src_eye_center = _get_eye_center(src_pts)
        dst_eye_center = _get_eye_center(dst_pts)
        dx = dst_eye_center[0] - src_eye_center[0]
        dy = dst_eye_center[1] - src_eye_center[1]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    warped = cv2.warpAffine(src_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped_pts = apply_affine_to_points(src_pts, M)
    
    # 改进的后处理：基于眼部中心进行精确对齐
    dst_eye_center = _get_eye_center(dst_pts)
    warped_eye_center = _get_eye_center(warped_pts)
    cx_err = float(dst_eye_center[0] - warped_eye_center[0])
    cy_err = float(dst_eye_center[1] - warped_eye_center[1])
    
    if abs(cx_err) > 1.0 or abs(cy_err) > 1.0:  # 降低阈值，提高精度
        M_corr = np.float32([[1, 0, cx_err], [0, 1, cy_err]])
        warped = cv2.warpAffine(warped, M_corr, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_pts = apply_affine_to_points(warped_pts, M_corr)
    
    return warped, warped_pts


def _make_improved_alignment_points(points: np.ndarray) -> np.ndarray:
    """改进的对齐点选择：基于眼睛、鼻子、嘴巴、下颌等关键面部特征点"""
    anchors = []
    
    # 定义关键面部特征点索引
    eye_points = {
        'right_outer': 33,   # 右眼外角
        'right_inner': 133,  # 右眼内角
        'left_inner': 362,   # 左眼内角
        'left_outer': 263,   # 左眼外角
    }
    
    nose_points = {
        'tip': 1,        # 鼻尖
        'bridge': 168,   # 鼻梁
        'left': 129,     # 左鼻翼
        'right': 358,    # 右鼻翼
    }
    
    mouth_points = {
        'left': 61,      # 左嘴角
        'right': 291,    # 右嘴角
        'top': 13,       # 上唇中心
        'bottom': 14,    # 下唇中心
    }
    
    jaw_points = {
        'chin': 199,     # 下巴中心
        'left': 132,     # 左下颌
        'right': 361,    # 右下颌
    }
    
    # 1. 眼睛关键点（最重要）
    valid_eye_points = []
    for name, idx in eye_points.items():
        if idx < len(points):
            valid_eye_points.append(points[idx])
    
    if len(valid_eye_points) >= 4:
        # 添加所有眼部关键点
        anchors.extend(valid_eye_points)
        # 眼部中心点
        right_center = (valid_eye_points[0] + valid_eye_points[1]) / 2.0  # 右眼中心
        left_center = (valid_eye_points[2] + valid_eye_points[3]) / 2.0   # 左眼中心
        anchors.append(right_center)
        anchors.append(left_center)
        anchors.append((right_center + left_center) / 2.0)  # 双眼中心
    elif EYER_OUTER < len(points) and EYEL_OUTER < len(points):
        # 回退到原来的方法
        anchors.append(points[EYER_OUTER])
        anchors.append(points[EYEL_OUTER])
        anchors.append((points[EYER_OUTER] + points[EYEL_OUTER]) / 2.0)
    
    # 2. 鼻子关键点
    valid_nose_points = []
    for name, idx in nose_points.items():
        if idx < len(points):
            valid_nose_points.append(points[idx])
    
    if len(valid_nose_points) >= 2:
        anchors.extend(valid_nose_points)
        # 鼻子中心
        if len(valid_nose_points) >= 4:
            nose_center = np.mean(valid_nose_points, axis=0)
            anchors.append(nose_center)
    
    # 3. 嘴巴关键点
    valid_mouth_points = []
    for name, idx in mouth_points.items():
        if idx < len(points):
            valid_mouth_points.append(points[idx])
    
    if len(valid_mouth_points) >= 2:
        anchors.extend(valid_mouth_points)
        # 嘴巴中心
        if len(valid_mouth_points) >= 4:
            mouth_center = np.mean(valid_mouth_points, axis=0)
            anchors.append(mouth_center)
    
    # 4. 下颌关键点
    valid_jaw_points = []
    for name, idx in jaw_points.items():
        if idx < len(points):
            valid_jaw_points.append(points[idx])
    
    if len(valid_jaw_points) >= 2:
        anchors.extend(valid_jaw_points)
        # 下颌中心
        if len(valid_jaw_points) >= 3:
            jaw_center = np.mean(valid_jaw_points, axis=0)
            anchors.append(jaw_center)
    
    return np.array(anchors, dtype=np.float32)


def _get_eye_center(points: np.ndarray) -> Tuple[float, float]:
    """获取眼部中心坐标"""
    if EYER_OUTER < len(points) and EYEL_OUTER < len(points):
        pr = points[EYER_OUTER]
        pl = points[EYEL_OUTER]
        return ((pr[0] + pl[0]) / 2.0, (pr[1] + pl[1]) / 2.0)
    else:
        # 回退到面部中心
        return (points.mean(axis=0)[0], points.mean(axis=0)[1])
