import os
import sys

# === 动态添加模块搜索路径 ===
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    # 打包模式：PyInstaller 解压路径
    base_dir = sys._MEIPASS

    # 1. 添加根目录（让 utils.py、config.py 可导入）
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)

    # 2. 添加 src/ 目录（如果需要从 src 导入其他模块）
    src_path = os.path.join(base_dir, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)

else:
    # 开发模式：将当前文件所在目录（src）加入路径
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    if curr_dir not in sys.path:
        sys.path.insert(0, curr_dir)

    # 可选：将项目根目录（上一级）也加入，模拟打包环境
    project_root = os.path.dirname(curr_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

try:
	from src.webapp import start  # when 'src' is a top-level package in sys.path
except Exception:
	# Fallback when executed as a package module
	from src.webapp import start

if __name__ == "__main__":
	start()
