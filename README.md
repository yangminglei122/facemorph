# Face Growth Morph (人像成长渐变)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

一个完整的人像成长渐变项目，可以将一个人不同时间拍摄的多张照片生成平滑的渐变动画（GIF/MP4）。

![示例输出](output/morph.gif)

## 项目概述

Face Growth Morph 是一个功能完整的人像成长渐变生成工具，能够：

- 读取一个人不同时间拍摄的多张照片（按文件名或时间排序）
- 检测人物、人脸、人脸关键点
- 自动选择最正面的参考图
- 以人物为中心做旋转+平移+等比缩放（不做形变），对齐其它图像
- 生成全图平滑渐变（支持光流插帧与非线性权重曲线），导出 GIF 和 MP4
- 提供本地网页界面（支持上传文件/ZIP、显示进度与预计剩余时间、下载结果）

## 核心功能特性

- **人脸关键点检测**：使用 MediaPipe 精准检测人脸关键点
- **智能对齐**：基于人脸关键点的相似性变换对齐，保持面部特征一致性
- **多种渐变模式**：
  - 交叉淡化（Crossfade）：简单线性过渡
  - 光流渐变（Optical Flow）：基于光流场的自然过渡
- **灵活的输出格式**：支持导出 GIF 和 MP4 视频
- **GPU 加速**：支持 CUDA 加速光流计算（可选）
- **Web 界面**：提供友好的 Web 界面，支持文件上传和进度显示
- **多平台支持**：支持 Linux/macOS/Windows

## 技术架构概览

```text
facemorph/
├── src/                  # 核心源代码
│   ├── __main__.py       # 程序入口
│   ├── webapp.py         # Web 界面实现
│   ├── pipeline.py       # 处理流程核心
│   ├── detect.py         # 人脸检测
│   ├── align.py          # 图像对齐
│   ├── morph.py          # 渐变生成
│   └── exporter.py       # 输出导出
├── images/               # 示例输入图片
├── output/               # 输出结果
├── scripts/              # 构建脚本
├── requirements.txt      # 依赖列表
└── README.md             # 项目文档
```

### 核心处理流程

1. **图像输入**：读取指定目录中的图像文件
2. **人脸检测**：使用 MediaPipe 检测每张图像的人脸关键点
3. **参考图选择**：自动选择面部最正面的图像作为参考
4. **图像对齐**：将所有图像对齐到参考图的坐标系
5. **渐变生成**：生成平滑的过渡帧序列
6. **结果导出**：保存为 GIF 和 MP4 格式

## 安装和依赖说明

### 环境要求

- Python 3.9+（建议 3.10）
- Linux/macOS/Windows（开发在 Linux 验证）

### 安装步骤

1. 创建虚拟环境（推荐）：
```bash
conda create -n facemorph python=3.10 -y
conda activate facemorph
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

### 依赖库说明

- `mediapipe`：人脸关键点检测
- `opencv-python`：图像处理和光流计算
- `numpy`：数值计算
- `Pillow`：图像格式支持
- `imageio`：GIF 导出
- `imageio-ffmpeg`：MP4 导出
- `flask`：Web 界面
- `pyinstaller`：打包工具
- `gunicorn`：生产环境 Web 服务器

## 使用方式

### 命令行模式

```bash
python main.py \
  --input_dir /path/to/photos \
  --output_dir /path/to/output \
  --sort name \
  --width 1080 --height 1350 \
  --transition_seconds 1.2 \
  --hold_seconds 0.4 \
  --gif_fps 15 \
  --video_fps 30
```

#### 命令行参数说明

- `--input_dir`：输入图片文件夹路径
- `--output_dir`：输出文件夹路径
- `--sort`：排序方式（`name`/`exif`/`filename_date`/`name_numeric`）
- `--width`/`--height`：输出图像尺寸
- `--subject_scale`：主体缩放系数（<1 保留更多背景）
- `--transition_seconds`：相邻图片间的渐变时长
- `--hold_seconds`：每张图片的保持时长
- `--gif_fps`/`--video_fps`：GIF 和视频的帧率
- `--morph`：渐变方式（`crossfade` 或 `flow`）
- `--flow_strength`：光流形变强度（0.5-1.5）
- `--face_protect`：面部保护权重（0-1）
- `--sharpen`：输出锐化强度（0-0.5）
- `--easing`：渐变权重曲线（`linear` 或 `compressed_mid`）
- `--use_gpu`：启用 GPU 加速（需要 CUDA 支持）

### Web 界面模式（推荐）

启动 Web 服务：
```bash
python -m src
# 或生产模式：
gunicorn -w 2 -b 0.0.0.0:5000 'src.webapp:app'
```

在浏览器中打开：`http://127.0.0.1:5000`

#### Web 界面功能

- 填写"输入目录"为服务器本机可访问路径
- 或上传图片（多文件）/ ZIP（优先使用上传内容）
- 实时显示处理进度和预计剩余时间
- 结果预览和下载（GIF/MP4/ZIP）

## 示例和输出说明

### 输入示例

项目包含示例图片：
```
images/
├── age-1.jpg
├── age-2.jpg
├── age-3.jpg
└── age-4.jpg
```

### 输出示例

处理完成后将在输出目录生成：
```
output/
├── morph.gif     # 渐变 GIF 动画
├── morph.mp4     # 渐变 MP4 视频
└── aligned/      # 对齐后的静态帧（可选）
```

### 输出参数说明

- **GIF**：适合在社交媒体分享，文件较小
- **MP4**：高质量视频，支持更多播放器
- **对齐帧**：处理过程中的中间结果，可用于调试

## 技术细节和算法说明

### 人脸检测与关键点提取

使用 MediaPipe 的 FaceMesh 模型检测人脸的 468 个关键点，包括：
- 眼部轮廓
- 嘴部轮廓
- 鼻部关键点
- 面部轮廓

### 图像对齐算法

1. **参考图选择**：基于眼部水平度、嘴部对称性和瞳距选择最佳参考图
2. **相似性变换**：使用关键点计算旋转、平移和缩放参数
3. **图像变换**：应用仿射变换将所有图像对齐到统一坐标系

### 渐变生成算法

#### 交叉淡化（Crossfade）

简单的线性插值：
```
frame = image1 * (1 - α) + image2 * α
```

#### 光流渐变（Optical Flow）

1. **光流计算**：使用 Farneback 算法计算前后帧的光流场
2. **面部保护**：对面部区域应用保护权重，减少变形
3. **帧插值**：基于光流场进行像素重映射和混合

### 非线性权重曲线

支持压缩中段的权重曲线，使渐变更自然：
```
if s <= a:
    alpha = (s / a) ** p1 * a
elif s >= b:
    alpha = 1.0 - ((1.0 - s) / (1.0 - b)) ** p3 * (1.0 - b)
else:
    alpha = a + (s - a) * (b - a) / (b - a)
```

## GPU 加速支持

### CUDA 支持检查

```bash
python - <<'PY'
import cv2
print('CUDA devices:', getattr(cv2, 'cuda', None) is not None and cv2.cuda.getCudaEnabledDeviceCount())
PY
```

### 构建 CUDA 版 OpenCV

```bash
bash scripts/build_opencv_cuda.sh
```

启用 GPU 加速：在 CLI 或网页勾选 `--use_gpu`。检测不到 CUDA 将自动回退 CPU。

## 打包和部署

### 打包为独立应用

项目已修复打包问题，现在可以正常生成独立的可执行文件。打包后的应用包含所有依赖项，可在没有Python环境的机器上运行。

#### 详细的PyInstaller打包步骤

1. 确保已安装所有依赖：
```bash
pip install -r requirements.txt
```

2. 使用PyInstaller进行打包：

**所有平台：**
```bash
pyinstaller FaceGrowthApp.spec
```

打包配置说明：
- 使用 `FaceGrowthApp.spec` 配置文件进行打包，确保所有依赖正确包含
- 入口文件为 `src/__main__.py`
- 包含必要的数据文件：`src` 目录和核心模块文件
- 隐藏导入多个模块：`flask`, `imageio`, `imageio_ffmpeg`, `cv2`, `numpy`, `PIL`, `mediapipe`, `scipy`, `tqdm`
- 自动收集 `mediapipe` 和 `imageio_ffmpeg` 的所有依赖文件
- 启用 UPX 压缩以减小打包文件大小
- 禁用控制台窗口（GUI应用模式）
- 不使用图标文件

3. 打包完成后，可执行文件将位于 `dist/` 目录中。

#### 不同平台的打包注意事项

- **跨平台兼容性**：使用 `FaceGrowthApp.spec` 配置文件打包时，已自动处理不同平台的路径分隔符差异
- **macOS**：可能需要额外的权限设置，确保打包后的应用有访问摄像头和文件系统的权限
- **Windows**：打包后的exe文件可能会被杀毒软件误报，需要添加到白名单
- **所有平台**：打包过程可能需要较长时间，请耐心等待

#### 打包后的文件结构和运行方式

打包完成后，项目根目录下会生成以下文件结构：
```
dist/
└── FaceGrowthApp     # 可执行文件（Linux/macOS）
└── FaceGrowthApp.exe # 可执行文件（Windows）
```

运行方式：
- **Linux/macOS**：在终端中运行 `./dist/FaceGrowthApp`
- **Windows**：双击 `dist/FaceGrowthApp.exe` 或在命令行中运行

运行后，在浏览器中访问 `http://127.0.0.1:5000` 即可使用Web界面。

#### 故障排除和常见问题解答

1. **打包失败或运行时缺少模块**：
   - 确保使用了 `FaceGrowthApp.spec` 配置文件进行打包，该文件已正确配置所有必要的依赖项
   - 检查是否所有依赖都已正确安装
   - 如果缺少特定模块，可以在 `FaceGrowthApp.spec` 的 `hiddenimports` 列表中添加

2. **打包后的应用无法启动**：
   - 检查是否有杀毒软件阻止了应用运行
   - 在Windows上尝试以管理员身份运行
   - 确保应用有必要的文件系统访问权限
   - 如果是由于UPX压缩导致的问题，可以尝试在 `FaceGrowthApp.spec` 中设置 `upx=False`

3. **Web界面无法访问**：
   - 确认应用已成功启动（查看控制台输出）
   - 检查防火墙设置是否阻止了端口5000
   - 尝试使用 `localhost:5000` 替代 `127.0.0.1:5000`

4. **处理速度慢或卡顿**：
   - 首次运行时可能需要加载模型，属于正常现象
   - 确保系统有足够的内存和CPU资源
   - 考虑使用GPU加速（如果硬件支持）

#### 验证打包成功的测试方法

1. **基本功能测试**：
   - 运行打包后的可执行文件
   - 在浏览器中访问Web界面
   - 上传示例图片或指定输入目录
   - 检查是否能正常生成GIF和MP4文件

2. **配置文件测试**：
   - 验证 `FaceGrowthApp.spec` 配置文件是否正确包含了所有必要的依赖项
   - 检查打包后的应用是否能正常访问所有资源文件
   - 确认隐藏导入的模块是否都能正常工作

3. **跨平台兼容性测试**：
   - 在目标平台上运行打包后的应用
   - 验证所有功能是否正常工作
   - 检查文件路径和权限问题

4. **性能测试**：
   - 处理多张图片，观察内存和CPU使用情况
   - 比较打包前后应用的性能差异
   - 验证GPU加速是否正常工作（如果启用）
   - 检查UPX压缩对性能的影响

通过以上测试，可以确保打包后的应用在各种环境下都能正常运行。

## 故障排除

### 常见问题

1. **Windows 路径问题**：Windows 路径不可直接在 Linux 使用。跨机访问时请用"上传"或将文件拷贝到服务器本机路径。

2. **MP4 未生成**：可能是系统缺少编码器。项目已做多重回退；必要时安装 `ffmpeg` 或使用生成的 GIF/ZIP。

3. **进度不更新**：确认以单进程测试（`gunicorn -w 1 ...`）；多进程需确保 `jobs/` 可写、浏览器未缓存 `/progress`。

### 多进程兼容性

- 进度与结果按"作业ID"落盘（`jobs/*.json`）
- 前端持有 `job` 并轮询 `/progress?job=...`
- 支持 `gunicorn -w N` 多进程部署

## 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目。

### 开发指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

本项目采用 MIT 许可证。详情请见 [LICENSE](LICENSE) 文件。

---

© 2025 Face Growth Morph. 保留所有权利。
