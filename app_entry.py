#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceGrowthApp 统一入口点
可以通过 --useweb 参数来选择运行命令行版本还是Web版本
"""

import sys
import os
import argparse

def main():
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