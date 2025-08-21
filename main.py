import argparse
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from utils import list_images_sorted, ensure_dir
from detect import LandmarkDetector
from align import select_reference_index, build_similarity_reference, warp_with_similarity
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


def main() -> None:
    args = parse_args()

    image_paths = list_images_sorted(args.input_dir, sort=args.sort)
    if len(image_paths) < 2:
        raise SystemExit("至少需要两张图片")

    print(f"读取 {len(image_paths)} 张图片，排序方式: {args.sort}")
    images = load_images(image_paths)

    detector = LandmarkDetector()
    detections = []
    for img in tqdm(images, desc="检测人脸/人物"):
        detections.append(detector.detect(img))
    detector.close()

    face_points_list = [d.face_points for d in detections]
    ref_idx = select_reference_index(face_points_list)
    print(f"参考图索引: {ref_idx} -> {os.path.basename(image_paths[ref_idx])}")

    target_h, target_w = args.height, args.width

    ref_det = detections[ref_idx]
    if ref_det.face_points is None:
        raise SystemExit("参考图未检测到人脸关键点，无法继续")

    ref_img = images[ref_idx]
    ref_pts_target, M_ref_total = build_similarity_reference(ref_det.face_points, (target_h, target_w), subject_scale=args.subject_scale)
    ref_img_warped = cv2.warpAffine(ref_img, M_ref_total, (target_w, target_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    aligned_by_index: List[Optional[np.ndarray]] = [None] * len(images)
    aligned_points_by_index: List[Optional[np.ndarray]] = [None] * len(images)
    aligned_by_index[ref_idx] = ref_img_warped
    aligned_points_by_index[ref_idx] = ref_pts_target

    for i, (img, det) in enumerate(zip(images, detections)):
        if i == ref_idx:
            continue
        if det.face_points is None:
            print(f"跳过：第 {i} 张未检测到人脸关键点")
            continue
        warped_img, warped_pts = warp_with_similarity(img, det.face_points, ref_pts_target, (target_h, target_w))
        aligned_by_index[i] = warped_img
        aligned_points_by_index[i] = warped_pts

    aligned_images: List[np.ndarray] = [im for im in aligned_by_index if im is not None]
    aligned_points: List[Optional[np.ndarray]] = [pts for pts in aligned_points_by_index if pts is not None]

    if len(aligned_images) < 2:
        raise SystemExit("有效对齐图像不足两张，无法生成渐变")

    if args.morph == "flow":
        frames = list(
            generate_flow_morph_frames(
                aligned_images,
                aligned_points,
                fps=args.video_fps,
                transition_seconds=args.transition_seconds,
                hold_seconds=args.hold_seconds,
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
