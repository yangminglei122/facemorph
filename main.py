import argparse
import os
import datetime as dt
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils import list_images_sorted, ensure_dir, read_exif_datetime, parse_date_from_filename, get_file_mtime_dt
from detect import LandmarkDetector
from align import select_reference_index, build_similarity_reference, warp_with_similarity, process_external_reference_image
from morph import generate_crossfade_frames, generate_flow_morph_frames
from exporter import save_gif, save_mp4


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Face Growth Morph")
    p.add_argument("--input_dir", default="./images", help="输入图片文件夹")
    p.add_argument("--output_dir", default="./output", help="输出文件夹")
    p.add_argument("--sort", default="name", choices=["name", "exif", "filename_date", "name_numeric"], help="排序方式(name/time/exif/filename_date/name_numeric)")
    p.add_argument("--width", type=int, default=1080, help="输出宽度")
    p.add_argument("--height", type=int, default=1350, help="输出高度")
    p.add_argument("--subject_scale", type=float, default=0.55, help="主体缩放系数(<1保留更多背景，=1不缩放)")
    p.add_argument("--transition_seconds", type=float, default=0.4, help="相邻两张渐变时长")
    p.add_argument("--hold_seconds", type=float, default=0.7, help="每张保持时长")
    p.add_argument("--gif_fps", type=int, default=15, help="GIF 帧率")
    p.add_argument("--video_fps", type=int, default=30, help="视频帧率")
    p.add_argument("--morph", choices=["crossfade", "flow"], default="flow", help="渐变方式")
    p.add_argument("--flow_strength", type=float, default=0.9, help="光流形变强度(0.5-1.5)")
    p.add_argument("--face_protect", type=float, default=0.7, help="面部保护权重(0-1)")
    p.add_argument("--sharpen", type=float, default=0.2, help="输出锐化强度(0-0.5)")
    p.add_argument("--easing", choices=["linear", "compressed_mid"], default="compressed_mid", help="渐变权重曲线")
    p.add_argument("--ease_a", type=float, default=0.4, help="中段起点(0-1)")
    p.add_argument("--ease_b", type=float, default=0.6, help="中段终点(0-1)")
    p.add_argument("--ease_p1", type=float, default=2.5, help="起段幂指数(>1更慢)")
    p.add_argument("--ease_p3", type=float, default=0.6, help="末段幂指数(<1更快)")
    p.add_argument("--use_gpu", action="store_true", help="若检测到可用CUDA则使用GPU光流加速")
    p.add_argument("--save_aligned", action="store_true", help="是否保存对齐后的静态帧")
    p.add_argument("--ref_index", type=int, help="手动指定参考图索引(0-based，不指定则自动选择)")
    p.add_argument("--ref_image", type=str, help="外部参考图像路径(不参与最终效果生成)")
    return p.parse_args()


def load_images(paths: List[str]) -> List[np.ndarray]:
    images = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取图像: {p}")
        images.append(img)
    return images


def extract_timestamps(paths: List[str], sort: str) -> List[Optional[dt.datetime]]:
    """从图片路径中提取时间戳信息"""
    timestamps = []
    for path in paths:
        # 优先使用EXIF时间
        timestamp = read_exif_datetime(path)
        if timestamp is None:
            # 尝试从文件名解析时间
            timestamp = parse_date_from_filename(path)
        if timestamp is None:
            # 使用文件修改时间
            timestamp = get_file_mtime_dt(path)
        timestamps.append(timestamp)
    return timestamps


def _debug_angle_calculation(pts: np.ndarray) -> None:
    """调试角度计算，显示各个特征点的角度"""
    if pts is None or len(pts) < 468:
        print("  关键点数量不足，无法进行详细分析")
        return
    
    # 定义关键面部特征点索引
    eye_points = {
        'right_outer': 33,   # 右眼外角
        'right_inner': 133,  # 右眼内角
        'left_inner': 362,   # 左眼内角
        'left_outer': 263,   # 左眼外角
    }
    
    nose_points = {
        'left': 129,     # 左鼻翼
        'right': 358,    # 右鼻翼
    }
    
    mouth_points = {
        'left': 61,      # 左嘴角
        'right': 291,    # 右嘴角
    }
    
    jaw_points = {
        'left': 132,     # 左下颌
        'right': 361,    # 右下颌
    }
    
    angles = []
    
    # 1. 眼睛连线角度
    if all(idx < len(pts) for idx in eye_points.values()):
        # 外眼角连线
        right_outer = pts[eye_points['right_outer']]
        left_outer = pts[eye_points['left_outer']]
        dy = float(left_outer[1] - right_outer[1])
        dx = float(left_outer[0] - right_outer[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            angles.append(('眼睛外角连线', angle_deg))
            print(f"    眼睛外角连线角度: {angle_deg:.2f}°")
        
        # 内眼角连线
        right_inner = pts[eye_points['right_inner']]
        left_inner = pts[eye_points['left_inner']]
        dy = float(left_inner[1] - right_inner[1])
        dx = float(left_inner[0] - right_inner[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            angles.append(('眼睛内角连线', angle_deg))
            print(f"    眼睛内角连线角度: {angle_deg:.2f}°")
    
    # 2. 鼻子水平线角度
    if all(idx < len(pts) for idx in nose_points.values()):
        nose_left = pts[nose_points['left']]
        nose_right = pts[nose_points['right']]
        dy = float(nose_left[1] - nose_right[1])
        dx = float(nose_left[0] - nose_right[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            angles.append(('鼻子水平线', angle_deg))
            print(f"    鼻子水平线角度: {angle_deg:.2f}°")
    
    # 3. 嘴巴连线角度
    if all(idx < len(pts) for idx in mouth_points.values()):
        mouth_left = pts[mouth_points['left']]
        mouth_right = pts[mouth_points['right']]
        dy = float(mouth_left[1] - mouth_right[1])
        dx = float(mouth_left[0] - mouth_right[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            angles.append(('嘴巴连线', angle_deg))
            print(f"    嘴巴连线角度: {angle_deg:.2f}°")
    
    # 4. 下颌连线角度
    if all(idx < len(pts) for idx in jaw_points.values()):
        jaw_left = pts[jaw_points['left']]
        jaw_right = pts[jaw_points['right']]
        dy = float(jaw_left[1] - jaw_right[1])
        dx = float(jaw_left[0] - jaw_right[0])
        if abs(dx) > 1e-6:
            angle_rad = np.arctan2(dy, dx)
            angle_deg = float(np.degrees(angle_rad))
            angles.append(('下颌连线', angle_deg))
            print(f"    下颌连线角度: {angle_deg:.2f}°")
    
    if angles:
        # 计算加权中位数
        eye_angles = [angle for name, angle in angles if '眼睛' in name]
        other_angles = [angle for name, angle in angles if '眼睛' not in name]
        
        if eye_angles:
            eye_median = float(np.median(eye_angles))
            print(f"    眼睛角度中位数: {eye_median:.2f}°")
        
        if other_angles:
            other_median = float(np.median(other_angles))
            print(f"    其他特征角度中位数: {other_median:.2f}°")
        
        if eye_angles and other_angles:
            final_angle = eye_median * 0.7 + other_median * 0.3
            print(f"    加权最终角度: {final_angle:.2f}°")
        elif eye_angles:
            print(f"    最终角度: {eye_median:.2f}°")
        else:
            print(f"    最终角度: {other_median:.2f}°")
    else:
        print("    无法计算任何角度")


def main() -> None:
    args = parse_args()

    image_paths = list_images_sorted(args.input_dir, sort=args.sort)
    if len(image_paths) < 2:
        raise SystemExit("至少需要两张图片")

    print(f"读取 {len(image_paths)} 张图片，排序方式: {args.sort}")
    images = load_images(image_paths)
    
    # 提取时间戳信息
    timestamps = extract_timestamps(image_paths, args.sort)
    print(f"提取到 {len([t for t in timestamps if t is not None])} 个有效时间戳")

    detector = LandmarkDetector()
    detections = []
    for img in tqdm(images, desc="检测人脸/人物"):
        detections.append(detector.detect(img))
    
    face_points_list = [d.face_points for d in detections]
    
        # 处理参考图选择
    external_ref_points = None
    if args.ref_image:
        # 使用外部参考图像
        external_ref_points = process_external_reference_image(args.ref_image, detector)
    
    detector.close()
    
    if args.ref_image:
        if external_ref_points is not None:
            print(f"使用外部参考图像: {args.ref_image}")
            # 外部参考图像不参与最终效果生成，所以ref_idx设为-1
            ref_idx = -1
        else:
            print(f"外部参考图像处理失败，将使用序列中的图像")
            ref_idx = select_reference_index(face_points_list, args.ref_index)
    else:
        # 使用序列中的图像作为参考
        ref_idx = select_reference_index(face_points_list, args.ref_index)
    
    if ref_idx >= 0:
        print(f"参考图索引: {ref_idx} -> {os.path.basename(image_paths[ref_idx])}")
        # 显示所有图像的质量评分和旋转角度
        print("所有图像质量评分和旋转角度:")
        for i, pts in enumerate(face_points_list):
            if pts is not None and len(pts) >= 300:
                from align import _compute_face_quality_score
                from utils import compute_roll_degrees
                quality = _compute_face_quality_score(pts)
                roll_angle = compute_roll_degrees(pts)
                marker = " ← 当前参考图" if i == ref_idx else ""
                print(f"  图像 {i}: 质量 {quality:.3f}, 旋转角度 {roll_angle:.2f}°{marker}")
        
        # 显示参考图的详细角度信息
        if ref_idx >= 0 and ref_idx < len(face_points_list):
            ref_pts = face_points_list[ref_idx]
            if ref_pts is not None:
                print(f"\n参考图 {ref_idx} 详细角度分析:")
                _debug_angle_calculation(ref_pts)
    else:
        print("使用外部参考图像，不参与最终效果生成")

    target_h, target_w = args.height, args.width

    # 构建参考几何
    if ref_idx >= 0:
        # 使用序列中的图像作为参考
        ref_det = detections[ref_idx]
        if ref_det.face_points is None:
            ref_filename = os.path.basename(image_paths[ref_idx])
            raise SystemExit(f"参考图未检测到人脸关键点，无法继续。参考图文件: {ref_filename}")
        
        ref_img = images[ref_idx]
        ref_pts_target, M_ref_total = build_similarity_reference(ref_det.face_points, (target_h, target_w), subject_scale=args.subject_scale)
        ref_img_warped = cv2.warpAffine(ref_img, M_ref_total, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        # 使用外部参考图像
        if external_ref_points is None:
            raise SystemExit("外部参考图像处理失败，无法继续")
        
        ref_pts_target, M_ref_total = build_similarity_reference(external_ref_points, (target_h, target_w), subject_scale=args.subject_scale)
        # 外部参考图像不参与最终效果生成，所以ref_img_warped设为None
        ref_img_warped = None

    aligned_by_index: List[Optional[np.ndarray]] = [None] * len(images)
    aligned_points_by_index: List[Optional[np.ndarray]] = [None] * len(images)
    
    # 只有当参考图来自序列时才添加到对齐结果中
    if ref_idx >= 0:
        aligned_by_index[ref_idx] = ref_img_warped
        aligned_points_by_index[ref_idx] = ref_pts_target

    for i, (img, det) in enumerate(zip(images, detections)):
        # 跳过参考图（如果它是外部参考图，ref_idx为-1，所以不会跳过任何图像）
        if ref_idx >= 0 and i == ref_idx:
            continue
        if det.face_points is None:
            filename = os.path.basename(image_paths[i])
            print(f"跳过：第 {i} 张未检测到人脸关键点。文件: {filename}")
            continue
        warped_img, warped_pts = warp_with_similarity(img, det.face_points, ref_pts_target, (target_h, target_w))
        aligned_by_index[i] = warped_img
        aligned_points_by_index[i] = warped_pts

    aligned_images: List[np.ndarray] = [im for im in aligned_by_index if im is not None]
    aligned_points: List[Optional[np.ndarray]] = [pts for pts in aligned_points_by_index if pts is not None]

    if len(aligned_images) < 2:
        raise SystemExit("有效对齐图像不足两张，无法生成渐变")

    # 为对齐后的图像创建对应的时间戳列表
    aligned_timestamps = []
    for i, (img, det) in enumerate(zip(images, detections)):
        if det.face_points is not None:
            # 只有当图像被成功对齐时才添加时间戳
            if aligned_by_index[i] is not None:
                aligned_timestamps.append(timestamps[i])
    
    if args.morph == "flow":
        frames = list(
            generate_flow_morph_frames(
                aligned_images,
                aligned_points,
                fps=args.video_fps,
                transition_seconds=args.transition_seconds,
                hold_seconds=args.hold_seconds,
                timestamps=aligned_timestamps,
                flow_strength=args.flow_strength,
                face_protect=args.face_protect,
                sharpen_amount=args.sharpen,
                easing=("compressed_mid" if args.easing != "linear" else "linear"),
                ease_a=args.ease_a,
                ease_b=args.ease_b,
                ease_p1=args.ease_p1,
                ease_p3=args.ease_p3,
                use_gpu=args.use_gpu,
            )
        )
    else:
        frames = list(
            generate_crossfade_frames(
                aligned_images,
                fps=args.video_fps,
                transition_seconds=args.transition_seconds,
                hold_seconds=args.hold_seconds,
                timestamps=aligned_timestamps,
                easing=("compressed_mid" if args.easing != "linear" else "linear"),
                ease_a=args.ease_a,
                ease_b=args.ease_b,
                ease_p1=args.ease_p1,
                ease_p3=args.ease_p3,
            )
        )

    ensure_dir(args.output_dir)
    gif_path = os.path.join(args.output_dir, "morph.gif")
    mp4_path = os.path.join(args.output_dir, "morph.mp4")

    print("保存 GIF...")
    save_gif(frames, gif_path, fps=args.gif_fps)
    print("保存 MP4...")
    save_mp4(frames, mp4_path, fps=args.video_fps)

    if args.save_aligned:
        aligned_dir = os.path.join(args.output_dir, "aligned")
        ensure_dir(aligned_dir)
        for idx, im in enumerate(aligned_images):
            cv2.imwrite(os.path.join(aligned_dir, f"aligned_{idx:03d}.png"), im)

    print(f"完成。GIF: {gif_path}  MP4: {mp4_path}")


if __name__ == "__main__":
    main()
