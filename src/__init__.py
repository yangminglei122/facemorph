import os

# Suppress noisy logs from GLOG/TF/absl before importing mediapipe
os.environ.setdefault("GLOG_minloglevel", "2")  # 0=INFO,1=WARNING,2=ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 2=ERROR
try:
	from absl import logging as absl_logging
	absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
	pass

__all__ = []
