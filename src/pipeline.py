from typing import Dict, Any, List, Optional, Tuple, Callable
import os
import cv2
import numpy as np
import datetime as dt
import logging

# 配置logger
logger = logging.getLogger(__name__)

# 修复导入路径问题，确保在PyInstaller打包环境中能正确导入
try:
    # 尝试相对导入（开发环境）
    from utils import list_images_sorted, ensure_dir, read_exif_datetime, parse_date_from_filename, get_file_mtime_dt, compute_roll_degrees
    from detect import LandmarkDetector
    from align import select_reference_index, build_similarity_reference, warp_with_similarity, process_external_reference_image
    from morph import generate_crossfade_frames, generate_flow_morph_frames
    from exporter import save_gif, save_mp4
except ImportError:
    # 在PyInstaller打包环境中，尝试从项目根目录导入
    import sys
    import os
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 添加项目根目录到sys.path
        root_dir = sys._MEIPASS
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
    from utils import list_images_sorted, ensure_dir, read_exif_datetime, parse_date_from_filename, get_file_mtime_dt, compute_roll_degrees
    from detect import LandmarkDetector
    from align import select_reference_index, build_similarity_reference, warp_with_similarity, process_external_reference_image
    from morph import generate_crossfade_frames, generate_flow_morph_frames
    from exporter import save_gif, save_mp4

ProgressCallback = Callable[[float, str], None]


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


def run_pipeline(config: Dict[str, Any], progress_cb: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    def report(p: float, msg: str):
        if progress_cb is not None:
            try:
                progress_cb(max(0.0, min(1.0, float(p))), str(msg))
            except Exception:
                pass

    try:
        # sanitize text inputs
        input_dir: str = str(config.get("input_dir", "./images")).strip()
        output_dir: str = str(config.get("output_dir", "./output")).strip()
        sort: str = str(config.get("sort", "name")).strip()
        width: int = int(config.get("width", 1080))
        height: int = int(config.get("height", 1350))
        subject_scale: float = float(config.get("subject_scale", 0.9))
        transition_seconds: float = float(config.get("transition_seconds", 1.0))
        hold_seconds: float = float(config.get("hold_seconds", 0.4))
        gif_fps: int = int(config.get("gif_fps", 15))
        video_fps: int = int(config.get("video_fps", 30))
        morph: str = str(config.get("morph", "flow")).strip()
        flow_strength: float = float(config.get("flow_strength", 0.9))
        face_protect: float = float(config.get("face_protect", 0.7))
        sharpen: float = float(config.get("sharpen", 0.2))
        easing: str = str(config.get("easing", "compressed_mid")).strip()
        ease_a: float = float(config.get("ease_a", 0.4))
        ease_b: float = float(config.get("ease_b", 0.6))
        ease_p1: float = float(config.get("ease_p1", 2.5))
        ease_p3: float = float(config.get("ease_p3", 0.6))
        use_gpu: bool = bool(config.get("use_gpu", False))
        save_aligned: bool = bool(config.get("save_aligned", False))

        # normalize output_dir to absolute to ensure download links are valid regardless of cwd
        output_dir_abs = os.path.abspath(output_dir)

        report(0.01, "扫描图片...")
        image_paths = list_images_sorted(input_dir, sort=sort)
        if len(image_paths) < 2:
            raise ValueError("至少需要两张图片")

        def _load_images(paths: List[str]) -> List[np.ndarray]:
            images = []
            for p in paths:
                img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    raise RuntimeError(f"无法读取图像: {p}")
                images.append(img)
            return images

        images = _load_images(image_paths)
        
        # 提取时间戳信息
        timestamps = extract_timestamps(image_paths, sort)
        report(0.015, f"提取到 {len([t for t in timestamps if t is not None])} 个有效时间戳")

        # Progress sections
        w_detect, w_align, w_morph, w_save = 0.30, 0.35, 0.30, 0.05

        # Detection
        report(0.02, "检测人脸/关键点...")
        detector = LandmarkDetector()
        detections: List = []
        N = len(images)
        for idx, img in enumerate(images):
            detections.append(detector.detect(img))
            report(0.02 + w_detect * (idx + 1) / max(1, N), f"检测 {idx+1}/{N}")
        detector.close()

        face_points_list = [d.face_points for d in detections]
        
        # 处理参考图选择（支持外部参考图和手动指定索引）
        manual_ref_idx = config.get("ref_index")
        ref_image_path = config.get("ref_image")  # 获取外部参考图路径
        
        # 记录每个图像的旋转角度
        logger.info("=== 图像旋转角度分析 ===")
        for idx, pts in enumerate(face_points_list):
            if pts is not None and len(pts) >= 300:
                roll_angle = compute_roll_degrees(pts)
                image_name = os.path.basename(image_paths[idx])
                logger.info(f"图像 {idx}: {image_name} - 旋转角度: {roll_angle:.2f}°")
            else:
                image_name = os.path.basename(image_paths[idx])
                logger.warning(f"图像 {idx}: {image_name} - 未检测到有效人脸关键点")
        
        # 处理外部参考图
        external_ref_points = None
        if ref_image_path and os.path.exists(ref_image_path):
            logger.info(f"检测到外部参考图: {ref_image_path}")
            try:
                # 创建新的检测器实例来处理外部参考图
                ref_detector = LandmarkDetector()
                external_ref_points = process_external_reference_image(ref_image_path, ref_detector)
                ref_detector.close()
                
                if external_ref_points is not None:
                    logger.info(f"外部参考图处理成功，检测到人脸关键点")
                    # 计算外部参考图的旋转角度
                    ref_roll_angle = compute_roll_degrees(external_ref_points)
                    logger.info(f"外部参考图旋转角度: {ref_roll_angle:.2f}°")
                else:
                    logger.warning(f"外部参考图处理失败，未检测到人脸关键点")
            except Exception as e:
                logger.error(f"处理外部参考图时出错: {e}")
                external_ref_points = None
        
        # 选择参考图索引
        if external_ref_points is not None:
            # 使用外部参考图
            logger.info("使用外部参考图进行对齐")
            ref_idx = -1  # 外部参考图不参与最终效果生成
        else:
            # 使用序列中的图像作为参考
            if manual_ref_idx is not None:
                try:
                    manual_ref_idx = int(manual_ref_idx)
                except (ValueError, TypeError):
                    manual_ref_idx = None
            
            ref_idx = select_reference_index(face_points_list, manual_ref_idx)
            if ref_idx >= 0:
                ref_image_name = os.path.basename(image_paths[ref_idx])
                logger.info(f"选择序列中的图像作为参考图: 索引 {ref_idx} - {ref_image_name}")
            else:
                logger.warning("无法选择有效的参考图，将使用第一张图像")
                ref_idx = 0

        target_h, target_w = height, width

        # 构建参考几何
        if ref_idx >= 0:
            # 使用序列中的图像作为参考
            ref_det = detections[ref_idx]
            if ref_det.face_points is None:
                ref_filename = os.path.basename(image_paths[ref_idx])
                raise ValueError(f"参考图未检测到人脸关键点，无法继续。参考图文件: {ref_filename}")

            # Build reference
            report(0.02 + w_detect, "构建参考几何...")
            ref_img = images[ref_idx]
            ref_pts_target, M_ref_total = build_similarity_reference(ref_det.face_points, (target_h, target_w), subject_scale=subject_scale)
            ref_img_warped = cv2.warpAffine(ref_img, M_ref_total, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            logger.info(f"序列参考图处理完成: {os.path.basename(image_paths[ref_idx])}")
        else:
            # 使用外部参考图像
            if external_ref_points is None:
                raise ValueError("外部参考图像处理失败，无法继续")
            
            logger.info("使用外部参考图构建参考几何")
            report(0.02 + w_detect, "构建参考几何（外部参考图）...")
            ref_pts_target, M_ref_total = build_similarity_reference(external_ref_points, (target_h, target_w), subject_scale=subject_scale)
            # 外部参考图像不参与最终效果生成，所以ref_img_warped设为None
            ref_img_warped = None
            logger.info(f"外部参考图处理完成: {ref_image_path}")

        # Align others
        aligned_by_index: List[Optional[np.ndarray]] = [None] * len(images)
        aligned_points_by_index: List[Optional[np.ndarray]] = [None] * len(images)
        
        # 只有当参考图来自序列时才添加到对齐结果中
        if ref_idx >= 0:
            aligned_by_index[ref_idx] = ref_img_warped
            aligned_points_by_index[ref_idx] = ref_pts_target
            logger.info(f"序列参考图已添加到对齐结果中: 索引 {ref_idx}")
        else:
            logger.info("使用外部参考图，序列参考图索引设为-1")

        report(0.02 + w_detect + 0.01, "对齐其他图像...")
        done = 0
        total_to_align = max(1, N - (1 if ref_idx >= 0 else 0))  # 如果使用外部参考图，总数为N
        
        for i, (img, det) in enumerate(zip(images, detections)):
            if ref_idx >= 0 and i == ref_idx:
                continue  # 跳过序列中的参考图
            if det.face_points is None:
                filename = os.path.basename(image_paths[i])
                report(0.02 + w_detect + w_align * (done / total_to_align), f"跳过：第 {i} 张未检测到人脸关键点。文件: {filename}")
                logger.warning(f"跳过图像 {i}: {filename} - 未检测到人脸关键点")
                continue
            
            logger.info(f"开始对齐图像 {i}: {os.path.basename(image_paths[i])}")
            warped_img, warped_pts = warp_with_similarity(img, det.face_points, ref_pts_target, (target_h, target_w))
            aligned_by_index[i] = warped_img
            aligned_points_by_index[i] = warped_pts
            done += 1
            report(0.02 + w_detect + w_align * (done / total_to_align), f"对齐 {done}/{total_to_align}")
            logger.info(f"图像 {i} 对齐完成: {os.path.basename(image_paths[i])}")

        aligned_images: List[np.ndarray] = [im for im in aligned_by_index if im is not None]
        aligned_points: List[Optional[np.ndarray]] = [pts for pts in aligned_points_by_index if pts is not None]

        logger.info(f"对齐完成，有效图像数量: {len(aligned_images)}/{len(images)}")
        
        if len(aligned_images) < 2:
            raise ValueError("有效对齐图像不足两张，无法生成渐变")

        # 为对齐后的图像创建对应的时间戳列表
        aligned_timestamps = []
        for i, (img, det) in enumerate(zip(images, detections)):
            if det.face_points is not None:
                # 只有当图像被成功对齐时才添加时间戳
                if aligned_by_index[i] is not None:
                    aligned_timestamps.append(timestamps[i])
                    logger.debug(f"添加时间戳: 图像 {i} - {os.path.basename(image_paths[i])}")

        logger.info(f"时间戳处理完成，有效时间戳数量: {len(aligned_timestamps)}")

        # Morph frames
        report(0.02 + w_detect + w_align + 0.01, "生成过渡帧...")
        logger.info("开始生成过渡帧...")
        frames: List[np.ndarray] = []
        if morph == "flow":
            transitions = len(aligned_images) - 1
            expected = video_fps * transition_seconds * transitions + int(hold_seconds * video_fps * transitions)
            logger.info(f"使用光流渐变，过渡数量: {transitions}，预期帧数: {expected}")
            gen = generate_flow_morph_frames(
                aligned_images,
                aligned_points,
                fps=video_fps,
                transition_seconds=transition_seconds,
                hold_seconds=hold_seconds,
                timestamps=aligned_timestamps,
                flow_strength=flow_strength,
                face_protect=face_protect,
                sharpen_amount=sharpen,
                easing=("compressed_mid" if easing != "linear" else "linear"),
                ease_a=ease_a,
                ease_b=ease_b,
                ease_p1=ease_p1,
                ease_p3=ease_p3,
                use_gpu=use_gpu,
            )
            for f in gen:
                frames.append(f)
                ratio = min(1.0, len(frames) / max(1, expected))
                report(0.02 + w_detect + w_align + w_morph * ratio, "生成过渡帧...")
        else:
            expected = video_fps * transition_seconds * (len(aligned_images) - 1) + int(hold_seconds * video_fps * (len(aligned_images) - 1))
            logger.info(f"使用交叉淡化渐变，预期帧数: {expected}")
            gen = generate_crossfade_frames(
                aligned_images,
                fps=video_fps,
                transition_seconds=transition_seconds,
                hold_seconds=hold_seconds,
                timestamps=aligned_timestamps,
                easing=("compressed_mid" if easing != "linear" else "linear"),
                ease_a=ease_a,
                ease_b=ease_b,
                ease_p1=ease_p1,
                ease_p3=ease_p3,
            )
            for f in gen:
                frames.append(f)
                ratio = min(1.0, len(frames) / max(1, expected))
                report(0.02 + w_detect + w_align + w_morph * ratio, "生成过渡帧...")
        
        logger.info(f"过渡帧生成完成，总帧数: {len(frames)}")

        # Save outputs
        report(0.02 + w_detect + w_align + w_morph + 0.01, "保存输出...")
        logger.info("开始保存输出文件...")
        
        ensure_dir(output_dir_abs)
        
        # Save GIF
        gif_path = os.path.join(output_dir_abs, "morph.gif")
        logger.info(f"保存GIF文件: {gif_path}")
        save_gif(frames, gif_path, gif_fps)
        
        # Save MP4
        mp4_path = os.path.join(output_dir_abs, "morph.mp4")
        logger.info(f"保存MP4文件: {mp4_path}")
        save_mp4(frames, mp4_path, video_fps)
        
        # Save aligned frames if requested
        if save_aligned:
            aligned_dir = os.path.join(output_dir_abs, "aligned")
            ensure_dir(aligned_dir)
            logger.info(f"保存对齐帧到: {aligned_dir}")
            for i, img in enumerate(aligned_images):
                if img is not None:
                    aligned_path = os.path.join(aligned_dir, f"aligned_{i:03d}.jpg")
                    cv2.imwrite(aligned_path, img)
        
        logger.info("输出文件保存完成")
        
        return {
            "gif_path": gif_path,
            "mp4_path": mp4_path,
            "aligned_dir": os.path.join(output_dir_abs, "aligned") if save_aligned else None,
        }
    except Exception as e:
        report(0.999, f"错误: {e}")
        raise
