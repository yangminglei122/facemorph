import os
import sys

# Ensure 'src' package is importable when running as a PyInstaller one-file binary
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
	base = sys._MEIPASS  # type: ignore[attr-defined]
	src_path = os.path.join(base, "src")
	if os.path.isdir(src_path) and src_path not in sys.path:
		sys.path.insert(0, src_path)
else:
	# When running from source, ensure this file's directory (the 'src' folder) is on sys.path
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	if curr_dir not in sys.path:
		sys.path.insert(0, curr_dir)

try:
	from src.webapp import start  # when 'src' is a top-level package in sys.path
except Exception:
	# Fallback when executed as a package module
	from .webapp import start

if __name__ == "__main__":
	start()
