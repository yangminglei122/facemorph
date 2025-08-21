#!/usr/bin/env bash
set -euo pipefail

# Build and install OpenCV with CUDA into current Python venv
# Requirements: NVIDIA driver + CUDA toolkit installed on system

OPENCV_VERSION="4.10.0"

if ! command -v python &>/dev/null; then
  echo "python not found in PATH" >&2
  exit 1
fi

PY_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
PY_EXEC="$(command -v python)"
SITEPKG="$(python - <<'PY'
import site, sys
print(site.getsitepackages()[0])
PY
)"

WORK="$(mktemp -d)"
echo "Working dir: $WORK"
cd "$WORK"

git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv.git
# Optional contrib (can enable if needed)
# git clone --branch ${OPENCV_VERSION} --depth 1 https://github.com/opencv/opencv_contrib.git

mkdir -p build && cd build

cmake ../opencv \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PY_PREFIX} \
  -DBUILD_LIST=core,imgproc,highgui,video,videoio,calib3d,features2d,flann,photo \
  -DBUILD_opencv_python3=ON \
  -DBUILD_opencv_python2=OFF \
  -DOPENCV_GENERATE_PKGCONFIG=ON \
  -DPYTHON3_EXECUTABLE=${PY_EXEC} \
  -DPYTHON3_PACKAGES_PATH=${SITEPKG} \
  -DWITH_CUDA=ON \
  -DCUDA_FAST_MATH=ON \
  -DWITH_CUBLAS=ON \
  -DWITH_CUDNN=ON \
  -DWITH_TBB=ON \
  -DWITH_OPENGL=ON \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_DOCS=OFF

make -j"$(nproc)"
make install

python - <<'PY'
import cv2
print('Installed OpenCV:', cv2.__version__)
print('CUDA devices:', getattr(cv2, 'cuda', None) is not None and cv2.cuda.getCudaEnabledDeviceCount())
PY

echo "Done."
