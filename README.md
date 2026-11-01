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
```

## 目录

| 目录 | 内容 |
|---|---|
| `backend/` | FastAPI 服务、推理核心、DICOM、传统细胞计数、测试 |
| `frontend/` | 单页前端（无构建步骤） |
| `cpp_backend/` | Drogon + OpenCV DNN C++ 服务、推理基准程序 |
| `scripts/fetch_models.py` | 下载第三方模型并生成 `models/<名称>.json`（类别、指标、推荐阈值、许可） |
| `scripts/fetch_samples.py` | 下载并生成示例图片 / 视频和 `samples/manifest.json` |
| `samples/` | 每个菜单的示例素材 |
| `training/` | 可选：用自己的数据训练、导出、评估 |
| `scripts/benchmark.py` | 多后端延迟 / 吞吐对比 |
| `docs/DESIGN.md` | 设计方案与开发步骤 |

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

模型权重不放进仓库，用 `scripts/fetch_models.py` 下载。页面上每张示例图下方都标注了来源和许可。

## 免责声明

本项目是技术演示，模型未经临床验证，**不得用于任何临床诊断或治疗决策**。
