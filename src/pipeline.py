from typing import Dict, Any, List, Optional, Tuple, Callable
import os
import cv2
import numpy as np
import datetime as dt
import gc

# 修复导入路径问题，确保在PyInstaller打包环境中能正确导入
try:
    # 尝试相对导入（开发环境）
    from utils import list_images_sorted, ensure_dir, read_exif_datetime, parse_date_from_filename, get_file_mtime_dt
    from detect import LandmarkDetector
    from align import select_reference_index, build_similarity_reference, warp_with_similarity, process_external_reference_image
    from morph import generate_crossfade_frames, generate_flow_morph_frames
    from exporter import save_gif, save_mp4, save_outputs_parallel
except ImportError:
    # 在PyInstaller打包环境中，尝试从项目根目录导入
    import sys
    import os
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # 添加项目根目录到sys.path
        root_dir = sys._MEIPASS
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
    from utils import list_images_sorted, ensure_dir, read_exif_datetime, parse_date_from_filename, get_file_mtime_dt
    from detect import LandmarkDetector
    from align import select_reference_index, build_similarity_reference, warp_with_similarity, process_external_reference_image
    from morph import generate_crossfade_frames, generate_flow_morph_frames
    from exporter import save_gif, save_mp4, save_outputs_parallel

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


def _load_images(paths: List[str]) -> List[np.ndarray]:
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


def _get_memory_usage() -> float:
    """获取当前内存使用量（MB）"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # 转换为MB
    except ImportError:
        return 0.0


def _process_batch(batch_paths: List[str], batch_timestamps: List[Optional[dt.datetime]],
                 ref_pts_target: np.ndarray, config: Dict[str, Any], start_idx: int = 0) -> Tuple[List[np.ndarray], List[np.ndarray], List[Optional[dt.datetime]]]:
    """处理单个批次的图像"""
    from align import warp_with_similarity
    from detect import LandmarkDetector
    
    # 加载当前批次图像
    batch_images = _load_images(batch_paths)
    
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
    
    width = int(config.get("width", 1080))
    height = int(config.get("height", 1350))
    
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


def _run_pipeline_in_batches(config: Dict[str, Any], progress_cb: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """分批处理大量图像以避免内存不足"""
    def report(p: float, msg: str):
        if progress_cb is not None:
            try:
                progress_cb(max(0.0, min(1.0, float(p))), str(msg))
            except Exception:
                pass

    # 提取配置参数
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
    generate_gif: bool = bool(config.get("generate_gif", True))
    batch_size: int = int(config.get("batch_size", 50))

    # normalize output_dir to absolute to ensure download links are valid regardless of cwd
    output_dir_abs = os.path.abspath(output_dir)

    report(0.01, "扫描图片...")
    image_paths = list_images_sorted(input_dir, sort=sort)
    if len(image_paths) < 2:
        raise ValueError("至少需要两张图片")

    # 提取时间戳信息
    timestamps = extract_timestamps(image_paths, sort)
    report(0.015, f"提取到 {len([t for t in timestamps if t is not None])} 个有效时间戳")

    # Progress sections
    w_detect, w_align, w_morph, w_save = 0.30, 0.35, 0.30, 0.05

    # 处理参考图选择
    manual_ref_idx = config.get("ref_index")
    if manual_ref_idx is not None:
        try:
            manual_ref_idx = int(manual_ref_idx)
        except (ValueError, TypeError):
            manual_ref_idx = None

    # 检查是否使用外部参考图
    ref_image_path = config.get("ref_image")
    if ref_image_path and os.path.exists(ref_image_path):
        # 使用外部参考图
        external_ref_points = process_external_reference_image(ref_image_path, LandmarkDetector())
        if external_ref_points is not None:
            # 外部参考图不参与最终效果生成，所以ref_idx设为-1
            ref_idx = -1
        else:
            # 外部参考图处理失败，回退到自动选择或手动指定
            ref_idx = select_reference_index([], manual_ref_idx)  # 空列表占位
    else:
        # 先处理第一批次以确定参考图
        first_batch_paths = image_paths[:batch_size]
        first_batch_timestamps = timestamps[:batch_size]
        
        # 加载第一批次图像
        first_batch_images = _load_images(first_batch_paths)
        
        # 检测第一批次人脸关键点
        detector = LandmarkDetector()
        first_batch_detections = []
        for img in first_batch_images:
            first_batch_detections.append(detector.detect(img))
        detector.close()
        
        first_batch_face_points = [d.face_points for d in first_batch_detections]
        
        # 选择参考图
        ref_idx = select_reference_index(first_batch_face_points, manual_ref_idx)
        
        # 释放第一批次内存
        del first_batch_images
        del first_batch_detections
        del first_batch_face_points
        gc.collect()

    target_h, target_w = height, width

    # Build reference
    report(0.02, "构建参考几何...")
    if ref_idx >= 0:
        # 需要重新加载包含参考图的批次来构建参考几何
        ref_batch_start = (ref_idx // batch_size) * batch_size
        ref_batch_end = min(ref_batch_start + batch_size, len(image_paths))
        ref_batch_paths = image_paths[ref_batch_start:ref_batch_end]
        ref_batch_timestamps = timestamps[ref_batch_start:ref_batch_end]
        
        # 加载参考批次图像
        ref_batch_images = _load_images(ref_batch_paths)
        
        # 检测参考批次人脸关键点
        detector = LandmarkDetector()
        ref_batch_detections = []
        for img in ref_batch_images:
            ref_batch_detections.append(detector.detect(img))
        detector.close()
        
        ref_batch_face_points = [d.face_points for d in ref_batch_detections]
        
        # 获取参考图
        local_ref_idx = ref_idx - ref_batch_start
        ref_det = ref_batch_detections[local_ref_idx]
        if ref_det.face_points is None:
            ref_filename = os.path.basename(ref_batch_paths[local_ref_idx])
            raise ValueError(f"参考图未检测到人脸关键点，无法继续。参考图文件: {ref_filename}")
        
        ref_img = ref_batch_images[local_ref_idx]
        ref_pts_target, M_ref_total = build_similarity_reference(ref_det.face_points, (target_h, target_w), subject_scale=subject_scale)
        ref_img_warped = cv2.warpAffine(ref_img, M_ref_total, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # 释放参考批次内存
        del ref_batch_images
        del ref_batch_detections
        del ref_batch_face_points
        gc.collect()
    else:
        # 使用外部参考图像
        if ref_image_path and os.path.exists(ref_image_path):
            external_ref_points = process_external_reference_image(ref_image_path, LandmarkDetector())
            if external_ref_points is None:
                raise ValueError("外部参考图像处理失败，无法继续")
            
            ref_pts_target, M_ref_total = build_similarity_reference(external_ref_points, (target_h, target_w), subject_scale=subject_scale)
            # 外部参考图像不参与最终效果生成，所以ref_img_warped设为None
            ref_img_warped = None
        else:
            raise ValueError("未提供有效的外部参考图像")

    # 分批处理所有图像
    all_aligned_images = []
    all_aligned_points = []
    all_aligned_timestamps = []
    
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_paths = image_paths[start_idx:end_idx]
        batch_timestamps = timestamps[start_idx:end_idx]
        
        report(0.02 + (batch_idx / total_batches) * 0.6, f"处理批次 {batch_idx+1}/{total_batches}...")
        
        # 处理当前批次
        aligned_images, aligned_points, aligned_timestamps = _process_batch(
            batch_paths, batch_timestamps, ref_pts_target, config, start_idx
        )
        
        # 添加到结果列表
        all_aligned_images.extend(aligned_images)
        all_aligned_points.extend(aligned_points)
        all_aligned_timestamps.extend(aligned_timestamps)
        
        # 显示内存使用情况
        memory_usage = _get_memory_usage()
        if memory_usage > 0:
            report(0.02 + (batch_idx / total_batches) * 0.6, f"处理批次 {batch_idx+1}/{total_batches}... (内存: {memory_usage:.1f}MB)")
        
        # 强制垃圾回收
        gc.collect()
    
    if len(all_aligned_images) < 2:
        raise ValueError("有效对齐图像不足两张，无法生成渐变")

    # Morph frames
    report(0.62, "生成过渡帧...")
    frames: List[np.ndarray] = []
    if morph == "flow":
        transitions = len(all_aligned_images) - 1
        expected = video_fps * transition_seconds * transitions + int(hold_seconds * video_fps * transitions)
        gen = generate_flow_morph_frames(
            all_aligned_images,
            all_aligned_points,
            fps=video_fps,
            transition_seconds=transition_seconds,
            hold_seconds=hold_seconds,
            timestamps=all_aligned_timestamps,
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
            report(0.62 + 0.3 * ratio, "生成过渡帧...")
    else:
        expected = video_fps * transition_seconds * (len(all_aligned_images) - 1) + int(hold_seconds * video_fps * (len(all_aligned_images) - 1))
        gen = generate_crossfade_frames(
            all_aligned_images,
            fps=video_fps,
            transition_seconds=transition_seconds,
            hold_seconds=hold_seconds,
            timestamps=all_aligned_timestamps,
            easing=("compressed_mid" if easing != "linear" else "linear"),
            ease_a=ease_a,
            ease_b=ease_b,
            ease_p1=ease_p1,
            ease_p3=ease_p3,
        )
        for f in gen:
            frames.append(f)
            ratio = min(1.0, len(frames) / max(1, expected))
            report(0.62 + 0.3 * ratio, "生成过渡帧...")

    # Save outputs
    report(0.95, "保存结果...")
    ensure_dir(output_dir_abs)
    gif_path = os.path.join(output_dir_abs, "morph.gif")
    mp4_path = os.path.join(output_dir_abs, "morph.mp4")

    # 使用并行保存函数保存GIF和MP4
    try:
        if generate_gif:
            save_outputs_parallel(frames, gif_path, mp4_path, gif_fps=gif_fps, mp4_fps=video_fps, progress_cb=lambda p, msg: report(0.95 + 0.03 * p, msg))
        else:
            # 只保存MP4
            gif_path = None  # 不生成GIF时设为None
            save_mp4(frames, mp4_path, fps=video_fps)
    except Exception as e:
        print(f"保存文件时出错: {e}")
        # 回退到原来的保存方式
        if generate_gif:
            save_gif(frames, gif_path, fps=gif_fps)
        try:
            save_mp4(frames, mp4_path, fps=video_fps)
        except Exception:
            pass

    if save_aligned:
        aligned_dir = os.path.join(output_dir_abs, "aligned")
        ensure_dir(aligned_dir)
        for idx, im in enumerate(all_aligned_images):
            cv2.imwrite(os.path.join(aligned_dir, f"aligned_{idx:03d}.png"), im)

    report(1.0, "完成")

    return {
        "ref_index": ref_idx,
        "num_inputs": len(image_paths),
        "gif_path": gif_path if gif_path and os.path.exists(gif_path) else None,
        "mp4_path": mp4_path if mp4_path and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0 else None,
        "output_dir": output_dir_abs,
        "image_paths": image_paths,
    }


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
        generate_gif: bool = bool(config.get("generate_gif", True))
        batch_size: int = int(config.get("batch_size", 50))

        # normalize output_dir to absolute to ensure download links are valid regardless of cwd
        output_dir_abs = os.path.abspath(output_dir)

        report(0.01, "扫描图片...")
        image_paths = list_images_sorted(input_dir, sort=sort)
        if len(image_paths) < 2:
            raise ValueError("至少需要两张图片")

        # 检查是否需要分批处理
        if len(image_paths) > batch_size:
            report(0.01, f"图像数量较多({len(image_paths)}张)，将分批处理以避免内存不足")
            return _run_pipeline_in_batches(config, progress_cb)

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
        
        # 处理参考图选择
        manual_ref_idx = config.get("ref_index")
        if manual_ref_idx is not None:
            try:
                manual_ref_idx = int(manual_ref_idx)
            except (ValueError, TypeError):
                manual_ref_idx = None
        
        # 检查是否使用外部参考图
        ref_image_path = config.get("ref_image")
        if ref_image_path and os.path.exists(ref_image_path):
            # 使用外部参考图
            external_ref_points = process_external_reference_image(ref_image_path, LandmarkDetector())
            if external_ref_points is not None:
                # 外部参考图不参与最终效果生成，所以ref_idx设为-1
                ref_idx = -1
            else:
                # 外部参考图处理失败，回退到自动选择或手动指定
                ref_idx = select_reference_index(face_points_list, manual_ref_idx)
        else:
            # 使用自动选择或手动指定
            ref_idx = select_reference_index(face_points_list, manual_ref_idx)

        target_h, target_w = height, width

        # Build reference
        report(0.02 + w_detect, "构建参考几何...")
        if ref_idx >= 0:
            # 使用序列中的图像作为参考
            ref_det = detections[ref_idx]
            if ref_det.face_points is None:
                ref_filename = os.path.basename(image_paths[ref_idx])
                raise ValueError(f"参考图未检测到人脸关键点，无法继续。参考图文件: {ref_filename}")
            
            ref_img = images[ref_idx]
            ref_pts_target, M_ref_total = build_similarity_reference(ref_det.face_points, (target_h, target_w), subject_scale=subject_scale)
            ref_img_warped = cv2.warpAffine(ref_img, M_ref_total, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            # 使用外部参考图像
            if ref_image_path and os.path.exists(ref_image_path):
                external_ref_points = process_external_reference_image(ref_image_path, LandmarkDetector())
                if external_ref_points is None:
                    raise ValueError("外部参考图像处理失败，无法继续")
                
                ref_pts_target, M_ref_total = build_similarity_reference(external_ref_points, (target_h, target_w), subject_scale=subject_scale)
                # 外部参考图像不参与最终效果生成，所以ref_img_warped设为None
                ref_img_warped = None
            else:
                raise ValueError("未提供有效的外部参考图像")

        # Align others
        aligned_by_index: List[Optional[np.ndarray]] = [None] * len(images)
        aligned_points_by_index: List[Optional[np.ndarray]] = [None] * len(images)
        if ref_idx >= 0:
            aligned_by_index[ref_idx] = ref_img_warped
            aligned_points_by_index[ref_idx] = ref_pts_target
        
        report(0.02 + w_detect + 0.01, "对齐其他图像...")
        done = 0
        total_to_align = len(images) if ref_idx < 0 else (len(images) - 1)
        for i, (img, det) in enumerate(zip(images, detections)):
            if i == ref_idx:
                continue
            if det.face_points is None:
                filename = os.path.basename(image_paths[i])
                print(f"警告：图像 {i} 人脸检测或关键点检测失败，文件: {filename}")
                report(0.02 + w_detect + w_align * (done / total_to_align), f"跳过：第 {i} 张未检测到人脸关键点。文件: {filename}")
                continue
            try:
                warped_img, warped_pts = warp_with_similarity(img, det.face_points, ref_pts_target, (target_h, target_w))
                aligned_by_index[i] = warped_img
                aligned_points_by_index[i] = warped_pts
                done += 1
                report(0.02 + w_detect + w_align * (done / total_to_align), f"对齐 {done}/{total_to_align}")
            except Exception as e:
                filename = os.path.basename(image_paths[i])
                print(f"警告：图像 {i} 对齐失败，文件: {filename}, 错误: {e}")
                continue

        aligned_images: List[np.ndarray] = [im for im in aligned_by_index if im is not None]
        aligned_points: List[Optional[np.ndarray]] = [pts for pts in aligned_points_by_index if pts is not None]

        if len(aligned_images) < 2:
            raise ValueError("有效对齐图像不足两张，无法生成渐变")

        # 为对齐后的图像创建对应的时间戳列表
        aligned_timestamps = []
        for i, (img, det) in enumerate(zip(images, detections)):
            if det.face_points is not None:
                # 只有当图像被成功对齐时才添加时间戳
                if aligned_by_index[i] is not None:
                    aligned_timestamps.append(timestamps[i])

        # Morph frames
        report(0.02 + w_detect + w_align + 0.01, "生成过渡帧...")
        frames: List[np.ndarray] = []
        if morph == "flow":
            transitions = len(aligned_images) - 1
            expected = video_fps * transition_seconds * transitions + int(hold_seconds * video_fps * transitions)
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

        # Save outputs
        report(0.98, "保存结果...")
        ensure_dir(output_dir_abs)
        gif_path = os.path.join(output_dir_abs, "morph.gif")
        mp4_path = os.path.join(output_dir_abs, "morph.mp4")

        # 使用并行保存函数保存GIF和MP4
        try:
            if generate_gif:
                save_outputs_parallel(frames, gif_path, mp4_path, gif_fps=gif_fps, mp4_fps=video_fps, progress_cb=lambda p, msg: report(0.98 + 0.02 * p, msg))
            else:
                # 只保存MP4
                gif_path = None  # 不生成GIF时设为None
                save_mp4(frames, mp4_path, fps=video_fps)
        except Exception as e:
            print(f"保存文件时出错: {e}")
            # 回退到原来的保存方式
            if generate_gif:
                save_gif(frames, gif_path, fps=gif_fps)
            try:
                save_mp4(frames, mp4_path, fps=video_fps)
            except Exception:
                pass

        if save_aligned:
            aligned_dir = os.path.join(output_dir_abs, "aligned")
            ensure_dir(aligned_dir)
            for idx, im in enumerate(aligned_images):
                cv2.imwrite(os.path.join(aligned_dir, f"aligned_{idx:03d}.png"), im)

        report(1.0, "完成")

        return {
            "ref_index": ref_idx,
            "num_inputs": len(image_paths),
            "gif_path": gif_path if gif_path and os.path.exists(gif_path) else None,
            "mp4_path": mp4_path if mp4_path and os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0 else None,
            "output_dir": output_dir_abs,
            "image_paths": image_paths,
        }
    except Exception as e:
        report(0.999, f"错误: {e}")
        raise
