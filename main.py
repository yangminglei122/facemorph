import argparse
import os
import datetime as dt
from typing import List, Optional, Tuple
import gc

import cv2
import numpy as np
from tqdm import tqdm

from utils import list_images_sorted, ensure_dir, read_exif_datetime, parse_date_from_filename, get_file_mtime_dt, extract_timestamps, load_images, get_memory_usage, process_batch
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
    p.add_argument("--subject_scale", type=float, default=0.75, help="主体缩放系数(<1保留更多背景，=1不缩放)")
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
    p.add_argument("--batch_size", type=int, default=50, help="分批处理时的批次大小(默认50，图像数量超过此值时自动分批)")
    p.add_argument("--generate_gif", action="store_true", help="是否生成GIF文件(默认生成)")
    p.add_argument("--streaming_save", action="store_true", help="是否使用流式保存以减少内存使用(处理大量图像时推荐使用)")
    return p.parse_args()



def load_single_image(image_path: str) -> np.ndarray:
    """加载单张图像"""
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    return img




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
            # 使用向量平均法计算加权角度，以正确处理角度的周期性
            eye_rad = np.radians(eye_median)
            other_rad = np.radians(other_median)
            # 将角度转换为向量分量
            eye_x, eye_y = np.cos(eye_rad), np.sin(eye_rad)
            other_x, other_y = np.cos(other_rad), np.sin(other_rad)
            # 计算加权向量分量
            final_x = eye_x * 0.7 + other_x * 0.3
            final_y = eye_y * 0.7 + other_y * 0.3
            # 将向量转换回角度
            final_rad = np.arctan2(final_y, final_x)
            final_angle = np.degrees(final_rad)
            print(f"    加权最终角度: {final_angle:.2f}°")
        elif eye_angles:
            print(f"    最终角度: {eye_median:.2f}°")
        else:
            print(f"    最终角度: {other_median:.2f}°")
    else:
        print("    无法计算任何角度")


def process_images_normal(image_paths: List[str], args) -> None:
    """正常处理图像（图像数量较少时）"""
    print("使用正常处理模式")
    
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

    # 继续处理...
    process_alignment_and_morphing(images, face_points_list, ref_pts_target, M_ref_total, 
                                 ref_img_warped, ref_idx, timestamps, args)


def process_images_in_batches(image_paths: List[str], args, batch_size: int) -> None:
    """分批处理图像（图像数量较多时，避免内存不足）"""
    print("使用分批处理模式")
    
    # 提取时间戳信息
    timestamps = extract_timestamps(image_paths, args.sort)
    print(f"提取到 {len([t for t in timestamps if t is not None])} 个有效时间戳")
    
    # 首先处理参考图选择
    detector = LandmarkDetector()
    
    # 如果指定了外部参考图像
    external_ref_points = None
    if args.ref_image:
        external_ref_points = process_external_reference_image(args.ref_image, detector)
        if external_ref_points is not None:
            print(f"使用外部参考图像: {args.ref_image}")
            ref_idx = -1
        else:
            print(f"外部参考图像处理失败，将使用序列中的图像")
            ref_idx = None
    else:
        ref_idx = None
    
    # 如果使用序列中的图像作为参考，需要先确定参考图
    if ref_idx is None:
        print("确定参考图...")
        ref_idx = select_reference_from_batches(image_paths, detector, args.ref_index, batch_size)
    
    detector.close()
    
    if ref_idx >= 0:
        print(f"参考图索引: {ref_idx} -> {os.path.basename(image_paths[ref_idx])}")
    
    # 分批处理图像
    target_h, target_w = args.height, args.width
    
    # 构建参考几何
    if ref_idx >= 0:
        # 加载参考图像
        ref_img = load_single_image(image_paths[ref_idx])
        ref_det = LandmarkDetector()
        ref_detection = ref_det.detect(ref_img)
        ref_det.close()
        
        if ref_detection.face_points is None:
            ref_filename = os.path.basename(image_paths[ref_idx])
            raise SystemExit(f"参考图未检测到人脸关键点，无法继续。参考图文件: {ref_filename}")
        
        ref_pts_target, M_ref_total = build_similarity_reference(ref_detection.face_points, (target_h, target_w), subject_scale=args.subject_scale)
        ref_img_warped = cv2.warpAffine(ref_img, M_ref_total, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # 释放参考图像内存
        del ref_img
        del ref_detection
    else:
        # 使用外部参考图像
        if external_ref_points is None:
            raise SystemExit("外部参考图像处理失败，无法继续")
        
        ref_pts_target, M_ref_total = build_similarity_reference(external_ref_points, (target_h, target_w), subject_scale=args.subject_scale)
        ref_img_warped = None
    
    # 分批处理图像
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    all_aligned_images = []
    all_aligned_points = []
    all_aligned_timestamps = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"\n处理批次 {batch_idx + 1}/{total_batches} (图像 {start_idx}-{end_idx-1})")
        
        # 处理当前批次
        batch_images, batch_points, batch_timestamps = process_batch(
            batch_paths, timestamps[start_idx:end_idx], ref_pts_target, args
        )
        
        # 保存当前批次的结果
        all_aligned_images.extend(batch_images)
        all_aligned_points.extend(batch_points)
        all_aligned_timestamps.extend(batch_timestamps)
        
        # 释放当前批次的内存
        del batch_images
        del batch_points
        gc.collect()
        
        print(f"批次 {batch_idx + 1} 处理完成，当前内存使用: {get_memory_usage():.1f} MB")
    
    # 检查对齐后的图像数量
    if len(all_aligned_images) < 2:
        print(f"警告：对齐后有效图像数量不足（{len(all_aligned_images)}张），无法生成效果")
        # 创建一个空的结果目录
        os.makedirs(args.output_dir, exist_ok=True)
        print("处理完成！")
        return
    
    # 生成最终效果
    print("\n生成最终效果...")
    if args.morph == "flow":
        frames = generate_flow_morph_frames(
            all_aligned_images, all_aligned_points, args.video_fps,
            args.transition_seconds, args.hold_seconds, timestamps=all_aligned_timestamps,
            flow_strength=args.flow_strength, face_protect=args.face_protect,
            sharpen_amount=args.sharpen, easing=args.easing,
            ease_a=args.ease_a, ease_b=args.ease_b, ease_p1=args.ease_p1,
            ease_p3=args.ease_p3, use_gpu=args.use_gpu
        )
    else:
        frames = generate_crossfade_frames(
            all_aligned_images, args.video_fps, args.transition_seconds,
            args.hold_seconds, timestamps=all_aligned_timestamps,
            easing=args.easing, ease_a=args.ease_a, ease_b=args.ease_b,
            ease_p1=args.ease_p1, ease_p3=args.ease_p3
        )
    
    # 保存结果
    if args.streaming_save:
        save_results_streaming(frames, args.output_dir, args.gif_fps, args.video_fps, args.generate_gif)
    else:
        save_results(frames, args.output_dir, args.gif_fps, args.video_fps, args.generate_gif)


def select_reference_from_batches(image_paths: List[str], detector, manual_ref_idx: Optional[int], batch_size: int) -> int:
    """从分批处理的图像中选择参考图"""
    print("从分批图像中选择最佳参考图...")
    
    all_face_points = []
    all_qualities = []
    
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        
        print(f"  分析批次 {batch_idx + 1}/{total_batches} 中的图像质量...")
        
        # 检测当前批次
        batch_images = load_images(batch_paths)
        batch_detections = []
        for img in tqdm(batch_images, desc=f"批次 {batch_idx + 1} 检测"):
            batch_detections.append(detector.detect(img))
        
        # 计算质量评分
        for i, detection in enumerate(batch_detections):
            if detection.face_points is not None and len(detection.face_points) >= 300:
                from align import _compute_face_quality_score
                quality = _compute_face_quality_score(detection.face_points)
                all_face_points.append(detection.face_points)
                all_qualities.append((start_idx + i, quality))
        
        # 释放当前批次内存
        del batch_images
        del batch_detections
        gc.collect()
    
    if not all_qualities:
        raise SystemExit("未检测到任何有效的人脸关键点")
    
    # 选择参考图
    if manual_ref_idx is not None:
        if manual_ref_idx >= len(image_paths):
            raise SystemExit(f"指定的参考图索引 {manual_ref_idx} 超出范围")
        ref_idx = manual_ref_idx
        print(f"使用手动指定的参考图索引: {ref_idx}")
    else:
        # 自动选择质量最高的图像
        best_idx, best_quality = max(all_qualities, key=lambda x: x[1])
        ref_idx = best_idx
        print(f"自动选择质量最高的图像作为参考图: {ref_idx} (质量: {best_quality:.3f})")
    
    return ref_idx



def process_alignment_and_morphing(images: List[np.ndarray], face_points_list: List[Optional[np.ndarray]], 
                                 ref_pts_target: np.ndarray, M_ref_total: np.ndarray,
                                 ref_img_warped: Optional[np.ndarray], ref_idx: int,
                                 timestamps: List[Optional[dt.datetime]], args) -> None:
    """处理图像对齐和变形"""
    # 对齐图像
    aligned_by_index: List[Optional[np.ndarray]] = [None] * len(images)
    aligned_points_by_index: List[Optional[np.ndarray]] = [None] * len(images)
    aligned_timestamps: List[Optional[dt.datetime]] = []
    
    print("对齐图像到参考图...")
    for i, (img, pts) in enumerate(tqdm(zip(images, face_points_list), total=len(images))):
        if pts is not None and len(pts) >= 300:
            try:
                # 对齐到参考图
                result = warp_with_similarity(img, pts, ref_pts_target, (args.height, args.width))
                if result is not None:
                    aligned_img, aligned_pts = result
                    aligned_by_index[i] = aligned_img
                    aligned_points_by_index[i] = aligned_pts
                    aligned_timestamps.append(timestamps[i])
            except Exception as e:
                print(f"警告：图像 {i} 对齐失败: {e}")
                continue
        else:
            # 人脸检测或关键点检测失败
            print(f"警告：图像 {i} 人脸检测或关键点检测失败")
            continue
    
    # 过滤掉None值
    aligned_images = [img for img in aligned_by_index if img is not None]
    aligned_points = [pts for pts in aligned_points_by_index if pts is not None]
    
    if len(aligned_images) < 2:
        raise SystemExit("对齐后有效图像数量不足，无法生成效果")
    
    print(f"成功对齐 {len(aligned_images)} 张图像")
    
    # 生成变形效果
    print("生成变形效果...")
    if args.morph == "flow":
        frames = generate_flow_morph_frames(
            aligned_images, aligned_points, args.video_fps,
            args.transition_seconds, args.hold_seconds, timestamps=aligned_timestamps,
            flow_strength=args.flow_strength, face_protect=args.face_protect,
            sharpen_amount=args.sharpen, easing=args.easing,
            ease_a=args.ease_a, ease_b=args.ease_b, ease_p1=args.ease_p1,
            ease_p3=args.ease_p3, use_gpu=args.use_gpu
        )
    else:
        frames = generate_crossfade_frames(
            aligned_images, args.video_fps, args.transition_seconds,
            args.hold_seconds, timestamps=aligned_timestamps,
            easing=args.easing, ease_a=args.ease_a, ease_b=args.ease_b,
            ease_p1=args.ease_p1, ease_p3=args.ease_p3
        )
    
    # 保存结果
    if args.streaming_save:
        save_results_streaming(frames, args.output_dir, args.gif_fps, args.video_fps, args.generate_gif)
    else:
        save_results(frames, args.output_dir, args.gif_fps, args.video_fps, args.generate_gif)


def save_results(frames: List[np.ndarray], output_dir: str, gif_fps: int, video_fps: int, generate_gif: bool = True) -> None:
    """保存结果文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理生成器类型的数据
    if hasattr(frames, '__iter__') and not hasattr(frames, '__getitem__'):
        # 如果是生成器，需要先转换为列表，以避免在多次遍历时耗尽
        frames = list(frames)
    
    # 保存GIF（可选）
    if generate_gif:
        gif_path = os.path.join(output_dir, "morph.gif")
        print(f"保存GIF: {gif_path}")
        save_gif(frames, gif_path, gif_fps)
    
    # 保存MP4
    mp4_path = os.path.join(output_dir, "morph.mp4")
    print(f"保存MP4: {mp4_path}")
    save_mp4(frames, mp4_path, video_fps)
    
    print("处理完成！")


def save_results_streaming(frame_generator, output_dir: str, gif_fps: int, video_fps: int, generate_gif: bool = True) -> None:
    """流式保存结果文件，逐帧写入以减少内存使用"""
    from exporter import save_mp4_streaming, save_gif
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果需要生成GIF，我们需要先收集所有帧（因为GIF需要所有帧）
    if generate_gif:
        print("收集帧数据以生成GIF...")
        frames = list(frame_generator)
        print(f"收集到 {len(frames)} 帧数据")
        
        # 保存GIF
        gif_path = os.path.join(output_dir, "morph.gif")
        print(f"保存GIF: {gif_path}")
        save_gif(frames, gif_path, gif_fps)
        
        # 保存MP4（使用流式保存）
        mp4_path = os.path.join(output_dir, "morph.mp4")
        print(f"流式保存MP4: {mp4_path}")
        save_mp4_streaming(iter(frames), mp4_path, video_fps)
    else:
        # 只保存MP4，直接使用流式保存
        mp4_path = os.path.join(output_dir, "morph.mp4")
        print(f"流式保存MP4: {mp4_path}")
        save_mp4_streaming(frame_generator, mp4_path, video_fps)
    
    print("处理完成！")


def main() -> None:
    args = parse_args()

    image_paths = list_images_sorted(args.input_dir, sort=args.sort)
    if len(image_paths) < 2:
        raise SystemExit("至少需要两张图片")

    print(f"读取 {len(image_paths)} 张图片，排序方式: {args.sort}")
    
    # 内存优化：检查图像数量，如果过多则分批处理
    max_images_per_batch = args.batch_size
    if len(image_paths) > max_images_per_batch:
        print(f"警告：图像数量较多({len(image_paths)}张)，将分批处理以避免内存不足")
        print(f"每批处理 {max_images_per_batch} 张图片")
        
        # 分批处理
        process_images_in_batches(image_paths, args, max_images_per_batch)
        return
    
    # 正常处理（图像数量较少时）
    process_images_normal(image_paths, args)


if __name__ == "__main__":
    main()
