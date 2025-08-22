import os
import threading
from typing import List

import imageio
import numpy as np
import cv2


def _to_rgb_frames(frames: List[np.ndarray]) -> List[np.ndarray]:
    """优化的帧格式转换函数，使用向量化操作提高性能"""
    if not frames:
        return []
    
    # 批量处理所有帧
    frames_array = np.array(frames)
    
    # 数据类型标准化
    if frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)
    
    # 处理单通道图像
    if frames_array.ndim == 3 and frames_array.shape[-1] == 1:
        frames_array = np.repeat(frames_array, 3, axis=-1)
    elif frames_array.ndim == 3 and frames_array.shape[-1] == 2:
        # 2通道图像，添加第三个通道
        frames_array = np.concatenate([frames_array, frames_array[:, :, :1]], axis=-1)
    
    # 处理4通道图像（BGRA -> RGB）
    if frames_array.ndim == 4 and frames_array.shape[-1] == 4:
        # 批量转换 BGRA -> RGB
        frames_array = frames_array[:, :, :, [2, 1, 0]]  # BGR -> RGB
    
    # 处理3通道图像（BGR -> RGB）
    elif frames_array.ndim == 4 and frames_array.shape[-1] == 3:
        frames_array = frames_array[:, :, :, ::-1]  # BGR -> RGB
    
    # 确保形状正确
    if frames_array.ndim == 4:
        return [frames_array[i] for i in range(frames_array.shape[0])]
    else:
        return [frames_array]


def save_gif(frames: List[np.ndarray], path: str, fps: int = 15) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    duration = 1.0 / max(1, fps)
    # 处理生成器类型的数据
    if hasattr(frames, '__iter__') and not hasattr(frames, '__getitem__'):
        # 如果是生成器，需要先转换为列表
        frames = list(frames)
    
    # 检查帧数据是否为空
    if not frames:
        print(f"警告：没有帧数据可保存GIF，frames类型: {type(frames)}")
        return
    
    print(f"保存GIF，帧数: {len(frames)}, 第一帧类型: {type(frames[0])}")
    rgb_frames = _to_rgb_frames(frames)
    imageio.mimsave(path, rgb_frames, format="GIF", duration=duration)


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 分批处理帧以减少内存使用
    batch_size = 50  # 每批处理的帧数
    
    # 处理生成器类型的数据
    if hasattr(frames, '__iter__') and not hasattr(frames, '__getitem__'):
        # 如果是生成器，需要先转换为列表
        frames = list(frames)
    
    # 检查帧数据是否为空
    if not frames:
        print(f"警告：没有帧数据可保存MP4，frames类型: {type(frames)}")
        return
    
    # 获取帧的尺寸（假设所有帧尺寸相同）
    
    print(f"保存MP4，帧数: {len(frames)}, 第一帧类型: {type(frames[0])}")
    first_frame_rgb = _to_rgb_frames([frames[0]])[0]
    h, w = first_frame_rgb.shape[:2]
    
    # 尝试使用imageio分批写入
    try:
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            rgb_batch = _to_rgb_frames(batch)
            for frame in rgb_batch:
                writer.append_data(frame)
        writer.close()
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return
    except Exception as e:
        print(f"ImageIO保存失败: {e}")
        pass
    
    # 回退到OpenCV VideoWriter (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not vw.isOpened():
        # last resort: write .avi
        alt_path = os.path.splitext(path)[0] + ".avi"
        fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
        vw = cv2.VideoWriter(alt_path, fourcc2, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError("无法创建视频文件（OpenCV VideoWriter 失败）")
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            rgb_batch = _to_rgb_frames(batch)
            for fr in rgb_batch:
                bgr = fr[:, :, ::-1]
                vw.write(bgr)
        vw.release()
        return
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        rgb_batch = _to_rgb_frames(batch)
        for fr in rgb_batch:
            bgr = fr[:, :, ::-1]
            vw.write(bgr)
    vw.release()


def save_outputs_parallel(frames: List[np.ndarray], gif_path: str, mp4_path: str, gif_fps: int = 15, mp4_fps: int = 30, progress_cb=None, optimize_speed: bool = True) -> None:
    """并行保存GIF和MP4文件，提高保存速度
    
    Args:
        frames: 要保存的帧列表
        gif_path: GIF文件保存路径
        mp4_path: MP4文件保存路径
        gif_fps: GIF帧率
        mp4_fps: MP4帧率
        progress_cb: 进度回调函数
        optimize_speed: 是否优化速度（降低质量以提高速度）
    """
    if not frames:
        raise ValueError("没有帧数据可保存")
    
    if progress_cb:
        progress_cb(0.0, "准备保存文件...")
    
    # 根据优化设置调整参数
    if optimize_speed:
        gif_quality = 5  # 降低GIF质量以提高速度
        mp4_quality = 6  # 降低MP4质量以提高速度
        mp4_preset = "ultrafast"  # 使用最快的编码预设
    else:
        gif_quality = 10  # 标准GIF质量
        mp4_quality = 8   # 标准MP4质量
        mp4_preset = "medium"  # 使用平衡的编码预设
    
    # 定义保存函数
    def save_gif_thread():
        try:
            if progress_cb:
                progress_cb(0.3, "保存GIF文件...")
            duration = 1.0 / max(1, gif_fps)
            # 使用分批处理保存GIF
            batch_size = 50
            writer = imageio.get_writer(gif_path, format="GIF", duration=duration, optimize=True, quality=gif_quality)
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i+batch_size]
                rgb_batch = _to_rgb_frames(batch)
                for frame in rgb_batch:
                    writer.append_data(frame)
            writer.close()
            if progress_cb:
                progress_cb(0.6, "GIF保存完成")
        except Exception as e:
            print(f"GIF保存失败: {e}")
            raise
    
    def save_mp4_thread():
        try:
            if progress_cb:
                progress_cb(0.4, "保存MP4文件...")
            # 使用分批处理保存MP4
            batch_size = 50
            frames_local = frames  # 创建局部变量引用
            
            # 转换第一帧以获取尺寸
            # 处理生成器类型的数据
            if hasattr(frames_local, '__iter__') and not hasattr(frames_local, '__getitem__'):
                # 如果是生成器，需要先转换为列表
                frames_local = list(frames_local)
            
            # 检查帧数据是否为空
            if not frames_local:
                raise ValueError("没有帧数据可保存")
            
            first_frame_rgb = _to_rgb_frames([frames_local[0]])[0]
            h, w = first_frame_rgb.shape[:2]
            
            # 优先使用libx264编码器，带速度优化
            if optimize_speed:
                writer = imageio.get_writer(mp4_path, fps=mp4_fps, codec="libx264",
                                           quality=mp4_quality, preset=mp4_preset, macro_block_size=None)
            else:
                writer = imageio.get_writer(mp4_path, fps=mp4_fps, codec="libx264",
                                           quality=mp4_quality, macro_block_size=None)
            
            for i in range(0, len(frames_local), batch_size):
                batch = frames_local[i:i+batch_size]
                rgb_batch = _to_rgb_frames(batch)
                for frame in rgb_batch:
                    writer.append_data(frame)
            writer.close()
            
            if not (os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0):
                raise Exception("libx264编码失败")
            if progress_cb:
                progress_cb(0.7, "MP4保存完成")
        except Exception:
            try:
                if progress_cb:
                    progress_cb(0.45, "MP4编码器回退...")
                # 回退到默认编码器
                batch_size = 50
                frames_local = frames  # 创建局部变量引用
                # 处理生成器类型的数据
                if hasattr(frames_local, '__iter__') and not hasattr(frames_local, '__getitem__'):
                    # 如果是生成器，需要先转换为列表
                    frames_local = list(frames_local)
                
                # 检查帧数据是否为空
                if not frames_local:
                    raise ValueError("没有帧数据可保存")
                    
                writer = imageio.get_writer(mp4_path, fps=mp4_fps)
                for i in range(0, len(frames_local), batch_size):
                    batch = frames_local[i:i+batch_size]
                    rgb_batch = _to_rgb_frames(batch)
                    for frame in rgb_batch:
                        writer.append_data(frame)
                writer.close()
                
                if not (os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0):
                    raise Exception("默认编码器失败")
                if progress_cb:
                    progress_cb(0.7, "MP4保存完成")
            except Exception:
                if progress_cb:
                    progress_cb(0.5, "MP4使用OpenCV编码...")
                # 最后回退到OpenCV
                batch_size = 50
                frames_local = frames  # 创建局部变量引用
                # 处理生成器类型的数据
                if hasattr(frames_local, '__iter__') and not hasattr(frames_local, '__getitem__'):
                    # 如果是生成器，需要先转换为列表
                    frames_local = list(frames_local)
                
                # 检查帧数据是否为空
                if not frames_local:
                    raise ValueError("没有帧数据可保存")
                
                first_frame_rgb = _to_rgb_frames([frames_local[0]])[0]
                h, w = first_frame_rgb.shape[:2]
                
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vw = cv2.VideoWriter(mp4_path, fourcc, mp4_fps, (w, h))
                if not vw.isOpened():
                    # 最后尝试保存为AVI
                    if progress_cb:
                        progress_cb(0.55, "MP4保存为AVI格式...")
                    alt_path = os.path.splitext(mp4_path)[0] + ".avi"
                    fourcc2 = cv2.VideoWriter_fourcc(*"XVID")
                    vw = cv2.VideoWriter(alt_path, fourcc2, mp4_fps, (w, h))
                    if not vw.isOpened():
                        raise RuntimeError("无法创建视频文件")
                    for i in range(0, len(frames_local), batch_size):
                        batch = frames_local[i:i+batch_size]
                        rgb_batch = _to_rgb_frames(batch)
                        for fr in rgb_batch:
                            bgr = fr[:, :, ::-1]
                            vw.write(bgr)
                    vw.release()
                    if progress_cb:
                        progress_cb(0.7, "AVI保存完成")
                    return
                
                for i in range(0, len(frames_local), batch_size):
                    batch = frames_local[i:i+batch_size]
                    rgb_batch = _to_rgb_frames(batch)
                    for fr in rgb_batch:
                        bgr = fr[:, :, ::-1]
                        vw.write(bgr)
                vw.release()
                if progress_cb:
                    progress_cb(0.7, "MP4保存完成")
    
    if progress_cb:
        progress_cb(0.25, "启动并行保存线程...")
    
    # 启动并行线程
    gif_thread = threading.Thread(target=save_gif_thread)
    mp4_thread = threading.Thread(target=save_mp4_thread)
    
    gif_thread.start()
    mp4_thread.start()
    
    # 等待两个线程完成
    gif_thread.join()
    mp4_thread.join()
    
    if progress_cb:
        progress_cb(1.0, "所有文件保存完成")


def save_mp4_streaming(frame_generator, path: str, fps: int = 30, progress_cb=None) -> None:
    """流式保存MP4文件，逐帧写入以减少内存使用
    
    Args:
        frame_generator: 帧生成器
        path: MP4文件保存路径
        fps: MP4帧率
        progress_cb: 进度回调函数
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 初始化编码器
    writer = None
    frame_count = 0
    
    try:
        for frame in frame_generator:
            # 初始化writer（在获取到第一帧后）
            if writer is None:
                print(f"初始化MP4写入器，第一帧类型: {type(frame)}")
                first_frame_rgb = _to_rgb_frames([frame])[0]
                h, w = first_frame_rgb.shape[:2]
                writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=None)
            
            # 转换帧并写入
            rgb_frame = _to_rgb_frames([frame])[0]
            writer.append_data(rgb_frame)
            frame_count += 1
            
            # 进度回调
            if progress_cb and frame_count % 10 == 0:
                progress_cb(frame_count, f"已写入 {frame_count} 帧")
        
        # 关闭writer
        if writer is not None:
            writer.close()
            print(f"MP4保存完成，共写入 {frame_count} 帧")
        else:
            print("警告：没有帧数据可保存MP4")
            
    except Exception as e:
        print(f"流式保存MP4失败: {e}")
        # 尝试关闭writer
        if writer is not None:
            try:
                writer.close()
            except:
                pass
        raise
