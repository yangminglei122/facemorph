#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceGrowthApp 统一入口点
可以通过 --useweb 参数来选择运行命令行版本还是Web版本
"""

import sys
import os
import argparse

def show_combined_help():
    """显示app_entry.py和main.py的联合帮助信息"""
    # 显示app_entry.py的帮助信息
    print("FaceGrowthApp 统一入口点")
    print("=" * 50)
    print("\n[入口点参数]")
    entry_parser = argparse.ArgumentParser(
        description="FaceGrowthApp 统一入口点",
        prog="FaceGrowthApp",
        add_help=False
    )
    entry_parser.add_argument("--useweb", action="store_true", help="使用Web界面版本")
    entry_parser.print_help()
    
    # 显示main.py的帮助信息
    print("\n" + "=" * 50)
    print("[命令行版本参数]")
    main_parser = argparse.ArgumentParser(
        description="Face Growth Morph",
        prog="FaceGrowthApp",
        add_help=False
    )
    main_parser.add_argument("--input_dir", default="./images", help="输入图片文件夹")
    main_parser.add_argument("--output_dir", default="./output", help="输出文件夹")
    main_parser.add_argument("--sort", default="name", choices=["name", "exif", "filename_date", "name_numeric"], help="排序方式(name/time/exif/filename_date/name_numeric)")
    main_parser.add_argument("--width", type=int, default=1080, help="输出宽度")
    main_parser.add_argument("--height", type=int, default=1350, help="输出高度")
    main_parser.add_argument("--subject_scale", type=float, default=0.75, help="主体缩放系数(<1保留更多背景，=1不缩放)")
    main_parser.add_argument("--transition_seconds", type=float, default=0.4, help="相邻两张渐变时长")
    main_parser.add_argument("--hold_seconds", type=float, default=0.7, help="每张保持时长")
    main_parser.add_argument("--gif_fps", type=int, default=15, help="GIF 帧率")
    main_parser.add_argument("--video_fps", type=int, default=30, help="视频帧率")
    main_parser.add_argument("--morph", choices=["crossfade", "flow"], default="flow", help="渐变方式")
    main_parser.add_argument("--flow_strength", type=float, default=0.9, help="光流形变强度(0.5-1.5)")
    main_parser.add_argument("--face_protect", type=float, default=0.7, help="面部保护权重(0-1)")
    main_parser.add_argument("--sharpen", type=float, default=0.2, help="输出锐化强度(0-0.5)")
    main_parser.add_argument("--easing", choices=["linear", "compressed_mid"], default="compressed_mid", help="渐变权重曲线")
    main_parser.add_argument("--ease_a", type=float, default=0.4, help="中段起点(0-1)")
    main_parser.add_argument("--ease_b", type=float, default=0.6, help="中段终点(0-1)")
    main_parser.add_argument("--ease_p1", type=float, default=2.5, help="起段幂指数(>1更慢)")
    main_parser.add_argument("--ease_p3", type=float, default=0.6, help="末段幂指数(<1更快)")
    main_parser.add_argument("--use_gpu", action="store_true", help="若检测到可用CUDA则使用GPU光流加速")
    main_parser.add_argument("--save_aligned", action="store_true", help="是否保存对齐后的静态帧")
    main_parser.add_argument("--ref_index", type=int, help="手动指定参考图索引(0-based，不指定则自动选择)")
    main_parser.add_argument("--ref_image", type=str, help="外部参考图像路径(不参与最终效果生成)")
    main_parser.add_argument("--batch_size", type=int, default=50, help="分批处理时的批次大小(默认50，图像数量超过此值时自动分批)")
    main_parser.add_argument("--generate_gif", action="store_true", help="是否生成GIF文件(默认生成)")
    main_parser.add_argument("--streaming_save", action="store_true", help="是否使用流式保存以减少内存使用(处理大量图像时推荐使用)")
    
    main_parser.print_help()

def main():
    # 检查是否请求帮助信息
    if "-h" in sys.argv or "--help" in sys.argv:
        show_combined_help()
        sys.exit(0)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="FaceGrowthApp")
    parser.add_argument("--useweb", action="store_true", help="使用Web界面版本")
    args, unknown_args = parser.parse_known_args()
    
    if args.useweb:
        # 运行Web版本
        try:
            from src.webapp import start
            # 移除--useweb参数，避免传递给Web应用
            sys.argv = [sys.argv[0]] + unknown_args
            start()
        except ImportError as e:
            print(f"无法导入Web应用模块: {e}")
            sys.exit(1)
    else:
        # 运行命令行版本
        try:
            from main import main as cmd_main
            # 保持原有的参数
            cmd_main()
        except ImportError as e:
            print(f"无法导入命令行模块: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()