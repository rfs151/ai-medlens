# MedLens · 医疗影像 AI 检测演示

> ⚠ **仅供技术演示，不用于临床诊断。**

基于 YOLO 的医疗影像检测 Web 演示系统。**只做推理，不需要训练**：全部使用第三方公开的预训练模型，页面内置示例图片和视频，也可以上传本地文件。

页面按影像种类分成 5 个菜单，**每个菜单对应一个独立的模型**（每个模型只认识自己那类影像的目标，比如牙科模型不会识别血细胞）：

| 菜单 | 模型（`models/<名称>.onnx`） | 检测目标 | 示例素材 |
|---|---|---|---|
| X 光骨折 | `fracture` · YOLO11n | 骨折 | 3 张 X 光片 |
| 血细胞计数 | `blood-cell` · YOLO11n | 红细胞、血小板、5 类白细胞 | 3 张血涂片 + 1 段显微镜平移视频；可与传统 OpenCV 算法对比 |
| 脑 MRI 肿瘤 | `brain-tumor` · YOLO11n | 胶质瘤、脑膜瘤、垂体瘤 | 4 张 MRI（含 1 张无肿瘤对照） |
| 口腔全景片 | `dental` · YOLO11n | 龋齿、根尖周病变、阻生齿 | 3 张全景片 |
| 结肠镜息肉 | `polyp` · YOLOv8，Kvasir-SEG 训练 | 息肉 | 2 段内镜视频 + 4 张截帧 |

其他功能：

- **视频实时检测**：内镜视频 / 本地视频 / 摄像头，显示 FPS 和延迟
- **双后端**：Python（FastAPI + ONNX Runtime）和 C++（Drogon + OpenCV DNN）提供同一套接口，页面一键切换对比延迟
- **专业细节**：DICOM 读取与窗宽窗位、置信度阈值滑块（按模型卡推荐值初始化）、模型指标卡片、检测报告导出
- **3D 体渲染（C++ / OpenGL）**：脑胶质瘤 MRI 的 GPU 光线投射体渲染 + 三视图（MPR）。同一套 C++17 代码编译成桌面程序（GLFW + OpenGL 3.3）和网页版（Emscripten → WebAssembly + WebGL2），详见下文

<!-- TODO: 演示 GIF -->

## 快速开始

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt onnx
cd backend && pytest -q && cd ..             # 测试不依赖任何模型

python scripts/fetch_models.py blood-cell fracture dental   # 作者直接发布了 ONNX，只需下载
# 息肉和脑肿瘤模型只发布了 PyTorch .pt，需要用 Ultralytics 导出成 ONNX。
# 先装 CPU 版 PyTorch（约 200 MB；直接 pip install ultralytics 会拉几 GB 的 CUDA 版）：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics && python scripts/fetch_models.py polyp brain-tumor

cd backend && uvicorn app.main:app --port 8000         # 打开 http://localhost:8000
```

国内网络可在命令前加 `HF_ENDPOINT=https://hf-mirror.com`。示例素材已经在 `samples/` 里；需要重新生成时运行 `python scripts/fetch_samples.py`（需要 curl 和 ffmpeg）。

缺少某个模型时服务照常启动，对应菜单会标记"未安装"并提示安装命令。

> 加载 `.pt` 文件会执行其中的 Python 序列化代码，只应加载可信来源的权重。导出成 ONNX 之后，推理端不再需要 PyTorch。

Docker 部署：

```bash
docker compose up -d --build                  # Python 服务 :8000
docker compose --profile cpp up -d --build    # 加上 C++ 服务 :8081
```

网络慢时（例如国内），C++ 镜像可以让 apt 走国内源，只让 git 走代理：

```bash
docker build --network host -f docker/Dockerfile.cpp -t medlens-cpp \
  --build-arg APT_MIRROR=http://mirrors.aliyun.com --build-arg GIT_PROXY=http://127.0.0.1:7890 .
docker run -d --name medlens-cpp -p 8081:8081 -v $PWD/models:/app/models:ro medlens-cpp
```

C++ 后端、部署、性能测试，以及用自己的数据训练模型（可选）的步骤见 **[docs/DESIGN.md](docs/DESIGN.md)**。

## 架构

```
浏览器 (HTML/JS) ──HTTP / WebSocket──► FastAPI ──► ONNX Runtime ──► JSON 检测框 ──► 前端 canvas 画框
                 └──HTTP────────────► Drogon (C++) ──► OpenCV DNN ──┘
                         models/<name>.onnx + <name>.json（两个后端共用）

浏览器 ──加载──► gl/medlens_volume.wasm（C++ 编译）──► WebGL2 ──► 本地 GPU 渲染 3D 体数据
桌面   ──────── medlens_volume（同一套 C++）──────► OpenGL 3.3
```

## 目录

| 目录 | 内容 |
|---|---|
| `backend/` | FastAPI 服务、推理核心、DICOM、传统细胞计数、测试 |
| `frontend/` | 单页前端（无构建步骤） |
| `cpp_backend/` | Drogon + OpenCV DNN C++ 服务、推理基准程序 |
| `gl_viewer/` | C++ / OpenGL 3D 体渲染查看器（桌面 + WebAssembly），编译产物在 `frontend/gl/` |
| `scripts/fetch_volume.py` | 下载一例脑肿瘤 MRI 体数据并转换为 `samples/volume/*.mlvol.gz` |
| `scripts/fetch_models.py` | 下载第三方模型并生成 `models/<名称>.json`（类别、指标、推荐阈值、许可） |
| `scripts/fetch_samples.py` | 下载并生成示例图片 / 视频和 `samples/manifest.json` |
| `samples/` | 每个菜单的示例素材 |
| `training/` | 可选：用自己的数据训练、导出、评估 |
| `scripts/benchmark.py` | 多后端延迟 / 吞吐对比 |
| `docs/DESIGN.md` | 设计方案与开发步骤 |

## 3D 体渲染（C++ / OpenGL）

菜单"3D 体渲染"展示一例脑胶质瘤 MRI（BraTS，142×178×136，FLAIR / T1 增强 / T2 三个序列）和专家标注的肿瘤三个分区：

- **2×2 工作站布局**：横断面 / 冠状面 / 矢状面三视图（MPR）+ 3D 视图。点击切面移动十字光标，三个视图联动，3D 视图中用彩色轮廓线标出切面位置
- **GPU 光线投射**：片元着色器里逐像素求光线与体数据包围盒的交点，从前往后采样合成；1D 传输函数查表、中心差分梯度光照、按梯度强调组织边界、步长变化时的不透明度校正、抖动起点消除条纹、不透明后提前终止
- **两种模式**：体渲染（DVR）和最大密度投影（MIP），三种配色预设，窗宽窗位
- **剖切**：沿视线方向的解析剖切面，切面显示为不透明的斜切片
- **肿瘤分区**：标注在 CPU 端做 3×3×3 平滑后存成 RGBA 3D 纹理，三线性采样得到光滑的等值面；水肿画成半透明外壳，透出内部的肿瘤核心；三个分区可单独开关，并统计体积（ml）
- **渐进式渲染**：交互时 3D 视图降到 50% 分辨率和 2 倍步长渲染到 FBO，停止 0.25 秒后自动以全分辨率重绘；空闲时不发出任何 GL 调用

**跨平台设计**（`gl_viewer/src/`）：

| 文件 | 作用 |
|---|---|
| `gl_api.h/.cc` | 平台抽象：网页直接用 OpenGL ES 3.0；桌面在运行时通过 `glfwGetProcAddress` 加载 OpenGL 3.3 函数（自写加载器，不依赖 GLEW / glad） |
| `shaders.h` | GLSL 只写一份，限定在 GLSL 3.30 与 GLSL ES 3.00 的公共子集，编译时只替换 `#version` 头 |
| `gl_objects.h/.cc` | 纹理 / FBO / 着色器程序的 RAII 封装（只能移动、析构时释放） |
| `volume.h/.cc` | `.mlvol` 解析、肿瘤掩膜平滑、体积统计 |
| `viewer.h/.cc` | 与平台无关的查看器：布局、相机（四元数轨迹球）、交互、脏标记与渐进式渲染 |
| `main_native.cc` | 桌面入口：GLFW 窗口、键鼠、HiDPI；另有离屏截图 `--screenshot` 和基准测试 `--bench` 模式 |
| `main_web.cc` | 网页入口：WebGL2 上下文、鼠标 / 触摸 / 滚轮事件、requestAnimationFrame 循环、导出给页面的 C 接口 |

```bash
python scripts/fetch_volume.py                      # 下载约 9 MB（HTTP 分段读取 7.6 GB 的 tar 包中的两个文件）

# 桌面版（Ubuntu：apt install libglfw3-dev zlib1g-dev libgl-dev）
cmake -S gl_viewer -B gl_viewer/build && cmake --build gl_viewer/build
gl_viewer/build/medlens_volume samples/volume/brats_206.mlvol.gz
gl_viewer/build/medlens_volume samples/volume/brats_206.mlvol.gz --screenshot shot.png --layout 1 --clip 0

# 网页版（需要 Emscripten SDK），产物自动复制到 frontend/gl/
emcmake cmake -S gl_viewer -B gl_viewer/build-web && cmake --build gl_viewer/build-web
```

网页版编译产物（JS 58 KB + wasm 62 KB）已提交在 `frontend/gl/`，部署时不需要 Emscripten。
已在 Linux（Mesa）和 Chrome（WebGL2）上验证。Windows（MSVC + vcpkg：`glfw3`、`zlib`、`opengl-registry` 提供 `GL/glcorearb.h`）和 macOS 按标准 CMake 写法准备，尚未实机编译验证。

## 模型指标

以下指标抄自各模型的模型卡，是作者在自己验证集上的结果，不是本项目复测的。

**模型选型**：每个菜单先在示例素材上实测，再决定用哪个模型。几个结论：
- 息肉：最初选的 REAL-Colon YOLO11m 在示例视频（HyperKvasir）上只有约 1% 的帧能检出，原因是两者来自不同医院、不同内镜设备。换成 Kvasir-SEG 训练的 YOLOv8 后，约 50% 的帧能检出。Kvasir-SEG 和 HyperKvasir 来自同一家医院。另外测过的 3 个候选：一个框太大、没有定位能力，一个实际是整图分类，一个效果更差。
- 脑肿瘤：在示例数据集的 27 张肿瘤图和 9 张正常图上实测，阈值 0.25 时正常图误报 5/9 张，所以默认阈值调到 0.35（检出 21/27，误报 3/9）。另外 2 个候选：一个在正常图上大量误报，另一个是分类模型。
- 医疗模型换一家医院、换一种设备，效果可能明显下降（domain shift）。这也是真实落地时最需要验证的地方。

| 模型 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 页面默认阈值 |
|---|---|---|---|---|---|
| blood-cell | 0.974 | 0.905 | 0.951 | 0.920 | 0.25 |
| fracture | 0.920 | 0.524 | 0.903 | 0.832 | 0.20 |
| brain-tumor | 0.961 | 0.667 | 0.883 | 0.932 | 0.35 |
| dental | 0.622 | 0.406 | 0.630 | 0.614 | 0.45（模型卡建议） |
| polyp | 模型卡未提供 | | | | 0.30 |

**两个后端的推理速度**（Intel i5-13500H，16 线程，CPU，单请求，各跑 12 次取中位数；检测结果两端完全一致）：

| 模型 | Python · ONNX Runtime | C++ · OpenCV DNN 4.10 |
|---|---|---|
| fracture（YOLO11n） | 36 ms | 105 ms |
| dental（YOLO11n） | 37 ms | 82 ms |
| polyp（YOLOv8，较大） | 107 ms | 183 ms |

在 CPU 上，OpenCV DNN 比 ONNX Runtime 慢 2–3 倍。所以 C++ 端的价值不在 OpenCV DNN 本身，而在于它提供了一个不依赖 Python、可以单独交付的推理服务框架。下一步提速可以把推理引擎换成 ONNX Runtime C++ API、OpenVINO，或者在 GPU 上用 TensorRT。

| 机器 | 后端 | p50 延迟 |
|---|---|---|
| GPU | ORT CUDA / TensorRT | 待填 |

## 模型、数据来源与许可

| 模型 | 来源 | 许可 |
|---|---|---|
| polyp | [pourmand1376/Kvasir-SEG-yolov8](https://huggingface.co/pourmand1376/Kvasir-SEG-yolov8) | MIT（权重）；训练数据 Kvasir-SEG 仅限研究与教学 |
| blood-cell | [Jesteban247/yolo11-blood_cell-onnx](https://huggingface.co/Jesteban247/yolo11-blood_cell-onnx) | Apache-2.0 |
| fracture | [Jesteban247/yolo11-fracture-onnx](https://huggingface.co/Jesteban247/yolo11-fracture-onnx) | MIT |
| brain-tumor | [amcm/brain-tumor-yolo11n](https://huggingface.co/amcm/brain-tumor-yolo11n) | AGPL-3.0（Ultralytics 基座） |
| dental | [liodon-ai/dental-panoramic-detector](https://huggingface.co/liodon-ai/dental-panoramic-detector) | **CC BY-NC 4.0，仅限非商业用途** |

| 示例素材（`samples/`） | 许可 |
|---|---|
| [HyperKvasir 息肉视频](https://huggingface.co/datasets/borgli/HyperKvasir-Polyp-Videos) — Borgli et al., Scientific Data 2020 | CC BY 4.0 |
| [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset) | MIT |
| [Roboflow-100 bone-fracture](https://huggingface.co/datasets/LibreYOLO/bone-fracture-7fylg) | CC BY 4.0 |
| [Brain Tumor MRI Dataset](https://huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset) — M. Nickparvar | CC BY 4.0 |
| [DENTEX](https://huggingface.co/datasets/tirandazdylan/dentex-panoramic) — Hamamci et al. | CC BY 4.0 |
| [Medical Segmentation Decathlon](http://medicaldecathlon.com/) Task01_BrainTumour（BraTS）— 3D 体渲染数据，`samples/volume/` 为裁剪并转换为 8 位后的衍生数据 | CC BY-SA 4.0 |

模型权重不放进仓库，用 `scripts/fetch_models.py` 下载。页面上每张示例图下方都标注了来源和许可。

## 免责声明

本项目是技术演示，模型未经临床验证，**不得用于任何临床诊断或治疗决策**。
