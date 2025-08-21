# Face Growth Morph (人像成长渐变)

一个完整的项目：
- 读取一个人不同时间拍摄的多张照片（按文件名或时间排序）
- 检测人物、人脸、人脸关键点
- 自动选择最正面的参考图
- 以人物为中心做旋转+平移+等比缩放（不做形变），对齐其它图像
- 生成全图平滑渐变（支持光流插帧与非线性权重曲线），导出 GIF 和 MP4
- 提供本地网页界面（支持上传文件/ZIP、显示进度与预计剩余时间、下载结果）

## 环境要求
- Python 3.9+（建议 3.10）
- Linux/macOS/Windows（开发在 Linux 验证）

## 安装
- 新建 conda 环境（推荐）：
```bash
conda create -n facemorph python=3.10 -y
conda activate facemorph
pip install -r requirements.txt
```
- 或在“当前环境”直接安装：
```bash
pip install -r requirements.txt
```

## 命令行用法（可选）
```bash
python -m src.main \
  --input_dir /path/to/photos \
  --output_dir /path/to/output \
  --sort name \
  --width 1080 --height 1350 \
  --transition_seconds 1.2 \
  --hold_seconds 0.4 \
  --gif_fps 15 \
  --video_fps 30
```
- `--sort`: `name`/`exif`/`filename_date`/`name_numeric`（`time` 同 `exif`，优先读 EXIF，失败则回退 mtime）
- `--subject_scale`: <1 可保留更多背景
- `--morph`: `flow` 或 `crossfade`
- 非线性权重：`--easing compressed_mid --ease_a 0.35 --ease_b 0.65 --ease_p1 2.4 --ease_p3 0.7`

## 网页界面（推荐）
- 启动：
```bash
python -m src
# 或生产模式：gunicorn -w 2 -b 0.0.0.0:5000 'src.webapp:app'
```
- 浏览器打开：`http://127.0.0.1:5000`
- 两种输入方式：
  - 填“输入目录”为服务器本机可访问路径（POSIX/本机文件系统）
  - 或上传图片（多文件）/ ZIP（优先使用上传内容）
- 页面会显示：
  - 进度条与阶段信息
  - 预计剩余时间（基于进度与已用时估算，保存阶段可能数秒到十数秒；显示仅供参考）
  - 结果预览与下载按钮（GIF/MP4/ZIP）。如 MP4 写入失败，仍可下载 GIF 与 ZIP。

多进程兼容
- 进度与结果按“作业ID”落盘（`jobs/*.json`），前端持有 `job` 并轮询 `/progress?job=...`，支持 `gunicorn -w N` 多进程部署。

## GPU 加速（可选，仅光流）
- 检查 CUDA：
```bash
python - <<'PY'
import cv2
print('CUDA devices:', getattr(cv2, 'cuda', None) is not None and cv2.cuda.getCudaEnabledDeviceCount())
PY
```
- 构建 CUDA 版 OpenCV（当前环境）：
```bash
bash scripts/build_opencv_cuda.sh
```
- 启用：在 CLI 或网页勾选 `--use_gpu`。检测不到 CUDA 将自动回退 CPU。

## 打包（可选）
- Linux/macOS：
```bash
pyinstaller -F -n FaceGrowthApp -c -w -i NONE --add-data "src:src" src/__main__.py
```
- Windows：
```bash
pyinstaller -F -n FaceGrowthApp -c -w -i NONE --add-data "src;src" src\__main__.py
```
- 若遇到缺包，可追加：
```bash
--collect-all mediapipe --collect-data imageio_ffmpeg
```
打包完成后运行 FaceGrowthApp，可在浏览器访问 `http://127.0.0.1:5000`。

## 排错
- Windows 路径不可直接在 Linux 使用。跨机访问时请用“上传”或将文件拷贝到服务器本机路径。
- MP4 未生成：可能是系统缺少编码器。项目已做多重回退；必要时安装 `ffmpeg` 或使用生成的 GIF/ZIP。
- 进度不更新/按钮不出现：确认以单进程测试（`gunicorn -w 1 ...`）；多进程需确保 `jobs/` 可写、浏览器未缓存 `/progress`（已禁用缓存）。

## 许可
MIT
