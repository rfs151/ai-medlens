# MedLens 设计方案与开发步骤

> 医疗影像 AI 检测 Web 演示系统：肠镜息肉实时检测（主项目） + 血细胞计数（传统算法 vs YOLO 对比，副项目）。
> **仅供技术演示，不用于临床诊断。**

---

## 1. 项目目标

| 目标 | 说明 | 验收标准 |
|---|---|---|
| 打开链接就能看 | 面试官浏览器打开即可上传图片 / 播放视频看到检测结果，不用装软件 | 公网 URL 可访问，首屏 < 2 s |
| 效果直观 | 内镜视频实时框出息肉，角落显示 FPS | CPU 机上 ≥ 10 FPS（320–480 px 输入、n/s 模型） |
| 展示工程深度 | 同一接口有 Python 和 C++ 两个后端，页面一键切换对比延迟 | 页面和 `scripts/benchmark.py` 都能给出两者的 p50/p95 |
| 体现业务理解 | 指标面板展示 mAP / Precision / Recall，阈值偏向高召回 | 有阈值扫描表和"推荐阈值" |
| 体现原理理解 | 血细胞：OpenCV 阈值 + 分水岭 vs YOLO 并排对比 | 同图两种结果和计数表 |
| 专业性 | 支持 DICOM 读取与窗宽窗位；报告导出；免责声明；数据许可 | 上传 .dcm 能显示并检测 |

不做的事：用户系统、数据库、病例管理、模型在线训练。这是演示系统，不是 PACS。

---

## 2. 总体架构

```
                         ┌──────────────────────── 浏览器 (frontend/, 纯 HTML/JS) ────────────────────────┐
                         │ 图片检测 │ 视频实时检测 │ 血细胞对比 │ 模型指标 │   [后端切换: Python / C++]    │
                         └────┬──────────────┬────────────────────────────────────────┬──────────────────┘
          HTTP multipart      │              │ WebSocket: JPEG 帧 ⇄ JSON 框             │ HTTP multipart
                              ▼              ▼                                          ▼
          ┌─────────────── Python 服务 :8000 (FastAPI) ───────────────┐     ┌──── C++ 服务 :8081 (Drogon) ────┐
          │ /api/detect  /api/detect/render  /ws/detect               │     │ /api/detect  /api/models        │
          │ /api/cells/classic (OpenCV 分水岭)   静态页面托管          │     │ DetectorPool (N 个 cv::dnn::Net) │
          │ imaging.py: JPG/PNG/DICOM(pydicom + 窗宽窗位)              │     │ letterbox → forward → NMS        │
          │ detector.py: letterbox → ONNX Runtime → decode → NMS       │     └──────────────┬───────────────────┘
          └──────────────────────────┬────────────────────────────────┘                    │
                                     └──────────── models/*.onnx + *.json (共享) ───────────┘
                                                              ▲
                     ┌────────────────── 训练流水线 (GPU 机, training/) ──────────────────┐
                     │ prepare_*.py → train.py (Ultralytics YOLO11) → export_onnx.py →   │
                     │ evaluate.py (mAP + 阈值扫描 → 写回 models/<name>.json)             │
                     └───────────────────────────────────────────────────────────────────┘
```

关键设计决策：

1. **模型文件是唯一契约**：`models/<name>.onnx` + `models/<name>.json`（类别名、输入尺寸、指标、推荐阈值、许可）。两个后端都从这里读，前端从 `/api/models` 读。新增模型 = 往目录里放两个文件，重启即可，不改代码。
2. **前端画框，后端只返回 JSON**：不传回结果图，省带宽；阈值滑块在前端本地过滤（后端以 conf=0.05 取一次全部候选框），拖动零延迟。IoU 改变才重新请求。
3. **视频在浏览器本地解码**：前端从 `<video>` 抓帧 → 缩放到 320/480/640 宽 → JPEG → WebSocket。服务器不需要 ffmpeg，也天然支持摄像头。**同时只有 1 帧在途**，网络或推理慢时自动跳帧，延迟不会累积。
4. **两个后端同一接口**：`POST /api/detect` 请求和响应格式完全一致（见第 5 节），前端只换 base URL。

---

## 3. 目录结构

```
ai-medlens/
├── backend/                  Python 服务
│   ├── app/
│   │   ├── main.py           FastAPI 路由、WebSocket、静态页面托管
│   │   ├── detector.py       YOLO ONNX 推理核心（letterbox / decode v5+v8 / NMS）
│   │   ├── registry.py       启动时加载 models/*.onnx
│   │   ├── imaging.py        图片 / DICOM 解码、窗宽窗位
│   │   ├── cells_classic.py  传统血细胞计数（阈值 + 距离变换 + 分水岭）
│   │   └── render.py         服务端画框（/api/detect/render）
│   ├── tests/                pytest，用构造的微型 ONNX 模型，不依赖训练结果
│   └── requirements*.txt
├── frontend/                 index.html / app.js / style.css（无构建步骤）
├── cpp_backend/              Drogon + OpenCV DNN 版本（CMake）
│   └── src/ main.cc  yolo_detector.{h,cc}  bench.cc
├── training/                 数据转换、训练、导出、评估 + data/*.yaml
├── scripts/benchmark.py      多后端延迟 / 吞吐对比
├── models/                   *.onnx（不入库）+ *.json（入库）
├── docker/                   Dockerfile.backend / Dockerfile.cpp
└── docker-compose.yml
```

---

## 4. 模块设计

### 4.1 数据与训练（`training/`）

| 数据集 | 用途 | 规模 | 转换 | 许可 |
|---|---|---|---|---|
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | 息肉检测 | 1000 张 + 掩膜 | `prepare_kvasir.py`：掩膜连通域 → 外接框（`--seg` 输出多边形，可训分割） | 仅限研究 / 教学 |
| [BCCD](https://github.com/Shenggan/BCCD_Dataset) | 血细胞检测 | 364 张，VOC XML | `prepare_bccd.py`：VOC → YOLO，沿用官方 train/val/test | MIT |
| [HyperKvasir](https://datasets.simula.no/hyper-kvasir/) 视频 | 视频演示素材 | 374 段内镜视频 | 直接用；或 `ffmpeg -framerate 10 -i %d.jpg demo.mp4` 把 val 图拼成视频 | CC BY 4.0 |

训练配置要点（`train.py`）：

- 模型：息肉用 `yolo11s`（精度/速度平衡），血细胞用 `yolo11n`（数据少，小模型不易过拟合）。CPU 部署要视频实时就用 n。
- 增强：上下 + 左右翻转（内镜和涂片没有固定方向）、小角度旋转、Mosaic；**色调扰动调小**（`hsv_h=0.01`），颜色本身带诊断信息。
- `patience=30` 早停；`batch=-1` 可以让 Ultralytics 按显存自动选。
- 数据划分固定随机种子，保证结果可复现。

评估（`evaluate.py`）：

- 输出 mAP@0.5、mAP@0.5:0.95、P、R 和逐类 AP，写回 `models/<name>.json`，页面"模型指标"页直接展示。
- **阈值扫描**：在 0.05–0.5 区间逐个阈值验证，取"召回率 ≥ 目标值（默认 0.90）时的最高阈值"作为推荐阈值。面试时说清楚：漏诊比误报代价高，误报可以由医生复核排除，所以阈值偏向高召回。

导出（`export_onnx.py`）：opset 12、静态 640 输入、`simplify=True`，同时写 sidecar JSON。OpenCV DNN 读不到 ONNX 自带的 metadata，所以类别名放在 JSON 里，两边共用。

### 4.2 推理核心（`backend/app/detector.py`）

```
BGR 图像 ──letterbox(灰边 114, 保持比例)──► RGB/255, NCHW float32 ──ORT──► 原始输出
   ──decode──► (cx,cy,w,h)→xyxy, 取最高类别分数, conf 过滤 ──按类 NMS(坐标偏移技巧)──► 去 letterbox 映射回原图
```

- 自动识别两种输出布局：YOLOv8/11 `(1, 4+nc, N)`，YOLOv5 `(1, N, 5+nc)`（分数 = objectness × 类别概率）。所以 v5/v8/11 的模型都能直接用。
- Execution Provider 优先级：TensorRT → CUDA → CPU，按机器上实际可用的选择。CPU 机装 `onnxruntime`，GPU 机装 `onnxruntime-gpu`，代码不用改。
- 返回各阶段耗时（预处理 / 推理 / 后处理），页面上直接展示，方便讲性能瓶颈在哪。
- `InferenceSession.run` 是线程安全的，FastAPI 的同步路由跑在线程池里，可以并发。

### 4.3 Python 服务（`backend/app/main.py`）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（Docker HEALTHCHECK 用） |
| GET | `/api/models` | 模型列表、类别、指标、推荐阈值 |
| POST | `/api/detect?model=&conf=&iou=&wc=&ww=` | multipart `file`，支持 JPG/PNG/DICOM，返回 JSON |
| POST | `/api/detect/render` | 同上，返回画好框的 PNG（curl 演示、无前端集成用） |
| POST | `/api/cells/classic` | 传统算法血细胞计数 |
| WS | `/ws/detect?model=&conf=` | 视频模式：二进制 JPEG 帧进，JSON 出；文本消息 `{"conf":0.3}` 可动态改参数 |

错误码：400 图片无法解码，404 模型不存在（消息里列出可用模型），413 文件超过 `MEDLENS_MAX_UPLOAD_MB`。

### 4.4 DICOM 支持（`imaging.py`）

- 通过扩展名或文件头第 128 字节的 `DICM` 标记识别。
- 处理顺序：`pixel_array` → `RescaleSlope/Intercept`（CT 转 HU 值）→ 窗宽窗位（请求参数优先，其次用 DICOM 标签，都没有就取全动态范围）→ `MONOCHROME1` 反色 → 8 位 BGR。
- 浏览器显示不了 DICOM，所以响应里附带 `image_png_base64`（加窗后的图）和 `dicom` 元信息，前端用它画图画框。
- 前端"DICOM 窗宽窗位"输入框改值就重新请求，可以现场演示调窗（如胸片肺窗 WC -600 / WW 1500，纵隔窗 WC 40 / WW 400）。

### 4.5 传统血细胞计数（`cells_classic.py`）

```
白细胞 / 血小板：HSV 中紫色、高饱和度 → 开运算去噪 → 闭运算合并核分叶 → 轮廓 → 按面积分（大 = WBC，小 = 血小板）
红细胞：         G 通道 Otsu 反阈值 → 扣掉紫色区域 → 开运算 → 填洞(红细胞中心淡)
                 → 距离变换 → 取峰值作为种子 → 分水岭分开粘连细胞 → 面积过滤 → 计数
```

面积阈值都按图像面积的比例设置，换分辨率也能用。演示要点：传统方法在细胞粘连、染色不均时会多算或少算，参数要按数据手调；YOLO 靠数据学习，泛化更好但需要标注。两者并排展示，说明你懂原理，不只是调包。

### 4.6 前端（`frontend/`）

纯 HTML/JS，不用构建工具，FastAPI 直接托管。四个标签页：

1. **图片检测**：拖拽 / 选择上传 → canvas 画框 → 右侧表格（类别、置信度、坐标）+ 类别计数柱状图 + 耗时拆分；阈值滑块本地过滤即时刷新；下载结果图；"导出报告"生成带免责声明的打印页，浏览器另存为 PDF（不需要后端 PDF 库）。
2. **视频实时检测**：本地视频或摄像头；`requestAnimationFrame` 循环叠加最近一次结果；HUD 显示检测 FPS、往返延迟、推理耗时、目标数；可选发送宽度（320/480/640/原始），现场演示分辨率对速度的影响。
3. **血细胞对比**：同一张图同时调 `/api/cells/classic` 和 bccd 模型，左右对比加计数表。
4. **模型指标**：mAP / P / R / 推荐阈值卡片 + 阈值扫描表 + 阈值策略说明。

后端切换：Python 用同源地址（视频走 WebSocket）；C++ 默认 `http://<当前主机>:8081`，也可以用 `?cpp=http://x:port` 指定（会记在 localStorage 里）。C++ 下视频模式改用 HTTP 逐帧请求。

### 4.7 C++ 后端（`cpp_backend/`）

- **Drogon**（异步高性能 HTTP 框架）+ **OpenCV DNN**（`readNetFromONNX`）。预处理、解码、NMS 的数学和 Python 版完全一致，结果可以对齐。
- `cv::dnn::Net::forward` 不是线程安全的 → `DetectorPool` 持有 N 个 Net 实例，请求借出、用完归还（RAII 归还，异常时也不会泄漏），满了就等待。N 由 `MEDLENS_POOL_SIZE` 控制。
- CORS：页面由 Python 服务托管，调用 C++ 属于跨域，用 pre-routing / post-handling advice 加响应头。
- `bench_detector`：不经过 HTTP 的纯推理基准（p50/p95/FPS），用来区分"网络 + 编解码开销"和"推理开销"。
- `MEDLENS_CUDA=1` 切 CUDA 后端（需要带 CUDA 编译的 OpenCV）。

---

## 5. 接口契约（两个后端一致）

```http
POST /api/detect?model=polyp&conf=0.25&iou=0.45
Content-Type: multipart/form-data; field "file"
```

```json
{
  "backend": "python-onnxruntime",          // 或 "cpp-drogon-opencv-dnn"
  "model": "polyp",
  "provider": "CPUExecutionProvider",
  "image": {"width": 622, "height": 529},
  "boxes": [{"cls": 0, "label": "polyp", "conf": 0.8731, "x1": 210.4, "y1": 118.0, "x2": 389.2, "y2": 301.7}],
  "counts": {"polyp": 1},
  "timing_ms": {"decode": 3.1, "preprocess": 1.9, "inference": 41.2, "postprocess": 0.6, "total": 43.7}
}
```

DICOM 请求额外返回 `dicom: {modality, window_center, window_width, rows, columns}` 和 `image_png_base64`。

`models/<name>.json`：

```json
{
  "names": ["polyp"], "imgsz": 640, "task": "detect",
  "description": "肠镜息肉检测 (Kvasir-SEG)",
  "license": "Kvasir-SEG: research & educational use only",
  "metrics": {"map50": 0.0, "map50_95": 0.0, "precision": 0.0, "recall": 0.0, "threshold_sweep": [...]},
  "recommended_conf": 0.15, "target_recall": 0.9
}
```

---

## 6. 性能设计与优化路线

| 层级 | 手段 | 预期 / 演示点 |
|---|---|---|
| 模型 | n / s 规模；输入 640 → 480/416 | 速度大约和像素数成正比 |
| 视频链路 | 前端降采样后再发；JPEG 质量 0.8；单帧在途、自动跳帧 | 延迟稳定，不会越来越卡 |
| ORT CPU | `ORT_ENABLE_ALL` 图优化；`MEDLENS_ORT_THREADS` 控制线程数 | 线程数 ≈ 物理核心数 |
| 量化 | `onnxruntime.quantization.quantize_dynamic` / 静态 INT8 | CPU 上快 1.5–2 倍，记录 mAP 损失 |
| GPU | `onnxruntime-gpu` CUDA EP；FP16 导出（`--half`） | 推理从几十 ms 降到几 ms |
| TensorRT | ORT TensorRT EP，或 `yolo export format=engine` + C++ TensorRT API | 演示 CPU ONNX vs GPU TensorRT 的速度对比 |
| C++ 服务 | Net 池并发；可改为"解码 / 推理 / 编码"多线程流水线 | 和 Python 对比同机 p50 / p95 / 吞吐 |

压测方法：

```bash
python scripts/benchmark.py samples/polyp.jpg --model polyp \
    --url py=http://localhost:8000 --url cpp=http://localhost:8081 -n 200 -c 4
./cpp_backend/build/bench_detector models/polyp.onnx samples/polyp.jpg 200
```

结果整理成表格放进 README：CPU 机（Vultr 普通型 / 阿里云）vs GPU 机，Python vs C++，ONNX FP32 vs INT8 vs TensorRT FP16。

---

## 7. 部署

**训练（GPU 机，按小时计费，用完就销毁）**

```bash
git clone <repo> && cd ai-medlens
pip install -r training/requirements.txt
# 下载数据 → prepare → train → evaluate → export（见第 8 节）
scp models/*.onnx models/*.json user@cpu-host:~/ai-medlens/models/   # 只带走模型文件
```

**服务（CPU 机长期运行）**

```bash
docker compose up -d --build                    # Python 服务 :8000
docker compose --profile cpp up -d --build      # 再加上 C++ 服务 :8081
```

**反向代理与 HTTPS（建议）**：浏览器只允许在 HTTPS 或 localhost 下调用摄像头（`getUserMedia`）。用 Nginx + Let's Encrypt，并记得转发 WebSocket：

```nginx
location /      { proxy_pass http://127.0.0.1:8000; client_max_body_size 30m; }
location /ws/   { proxy_pass http://127.0.0.1:8000; proxy_http_version 1.1;
                  proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade"; }
location /cpp/  { proxy_pass http://127.0.0.1:8081/; client_max_body_size 30m; }   # 页面用 ?cpp=https://域名/cpp
```

国内服务器绑域名需要先备案；临时演示用 `IP:8000` 也可以（这时只能用视频文件，不能用摄像头）。

---

## 8. 开发步骤（约 2 周，每步都有验收标准）

### 阶段 0：环境（0.5 天）

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cd backend && pytest -q                 # 8 个测试通过：不需要任何训练好的模型
uvicorn app.main:app --reload --port 8000   # 打开 http://localhost:8000
```

✅ 页面能打开，右上角显示"python-onnxruntime · 无模型"。

### 阶段 1：先打通全链路（0.5 天）

不训练，直接用 COCO 预训练模型验证"导出 → 加载 → 页面检测"整条链：

```bash
pip install -r training/requirements.txt
python training/export_onnx.py --weights yolo11n.pt --name coco-smoke
```

✅ 重启服务，上传任意生活照片，能框出 person / car 等。之后可以删掉 `models/coco-smoke.*`。

### 阶段 2：血细胞模型（1 天，CPU 也能训）

```bash
git clone https://github.com/Shenggan/BCCD_Dataset datasets/raw/BCCD_Dataset
python training/prepare_bccd.py --src datasets/raw/BCCD_Dataset --dst datasets/bccd
python training/train.py --data training/data/bccd.yaml --name bccd --model yolo11n.pt --epochs 80 --device 0
python training/export_onnx.py --weights runs/bccd/weights/best.pt --name bccd \
    --description "血细胞检测 (BCCD)" --license "BCCD Dataset, MIT"
python training/evaluate.py --weights runs/bccd/weights/best.pt --data training/data/bccd.yaml --name bccd
```

✅ "血细胞计数对比"页左右都有结果；"模型指标"页有 mAP 和阈值扫描表。根据实际图片微调 `cells_classic.py` 里的参数（`sat_thresh`、面积比例、`dist_ratio`）。

### 阶段 3：息肉模型（1–2 天，GPU 机）

```bash
# 下载 Kvasir-SEG.zip 解压到 datasets/raw/Kvasir-SEG
python training/prepare_kvasir.py --src datasets/raw/Kvasir-SEG --dst datasets/polyp
python training/train.py --data training/data/polyp.yaml --name polyp --model yolo11s.pt --epochs 100 --batch -1
python training/evaluate.py --weights runs/polyp/weights/best.pt --data training/data/polyp.yaml --name polyp
python training/export_onnx.py --weights runs/polyp/weights/best.pt --name polyp \
    --description "肠镜息肉检测 (Kvasir-SEG)" --license "Kvasir-SEG: research & educational use only"
```

> 注意：先 export 再 evaluate，或者两个都跑完再看 JSON——两个脚本都是合并写入同一个 `models/polyp.json`，不会互相覆盖。

✅ val mAP@0.5 达到 0.85 以上（Kvasir-SEG 上 YOLO 系列的常见水平）；页面上传内镜图片能框出息肉。若需 CPU 实时视频，再额外训一个 `yolo11n` 版本 `polyp-n`，两个模型都放进 `models/`，页面可以切换对比。

### 阶段 4：视频演示素材与调优（1 天）

1. 从 HyperKvasir 选 2–3 段有息肉的视频，剪成 20–30 秒（`ffmpeg -ss 00:01:00 -t 30 -i in.mp4 -vf scale=640:-2 demo.mp4`）。
2. 在 CPU 机上测试发送宽度 320 / 480 / 640 下的 FPS，记录下来。
3. 设定默认阈值 = `recommended_conf`。

✅ CPU 机上 n 模型 + 480 宽 ≥ 10 FPS；HUD 数值稳定，不会越来越慢。

### 阶段 5：C++ 后端（3–4 天）

```bash
# 本地（Ubuntu 需要 OpenCV ≥ 4.7，自带的 4.6 太旧，建议直接用 Docker）
docker compose --profile cpp up -d --build
curl -F file=@samples/polyp.jpg "http://localhost:8081/api/detect?model=polyp"
```

✅ 同一张图两个后端的框坐标差 < 1 px、置信度差 < 0.01（核对预处理是否一致）；页面切到 C++ 后图片和视频模式都能用；`benchmark.py` 输出对比表。

可选深入：`DetectorPool` 改成三段流水线（解码线程 / 推理线程 / 序列化线程）；ONNX Runtime C++ API 或 TensorRT 版本的 detector，用同一接口替换。

### 阶段 6：DICOM 与报告（0.5 天）

- 用 pydicom 自带示例或 VinBigData 的 DICOM 片子测试读取、调窗。
- 报告页检查：文件名、模型、阈值、时间、统计、结果图、明细表、免责声明。

✅ 上传 .dcm 能显示，改 WC/WW 后图像明暗变化。

### 阶段 7：部署与包装（1–2 天）

1. CPU 机 `docker compose up -d`，配 Nginx + HTTPS。
2. README：演示 GIF（用 ScreenToGif / peek 录图片检测、视频检测、阈值滑块、后端切换）、指标表、性能对比表、架构图、数据许可、免责声明。
3. 准备 3 张示例图 + 1 段示例视频放在 `samples/`（确认许可允许再分发；Kvasir-SEG 只能少量引用并注明出处）。

✅ 在别的电脑或手机上打开公网链接，完整走一遍演示流程。

---

## 9. 测试策略

| 层级 | 内容 | 位置 |
|---|---|---|
| 单元测试 | letterbox 几何、NMS、按类 NMS、v5/v8 解码、窗宽窗位 | `backend/tests` |
| 端到端（无需训练） | 用 `onnx.helper` 构造输出固定张量的微型模型，验证推理 + 坐标还原 + HTTP + WebSocket | `test_detector_end_to_end`、`test_api` |
| 算法回归 | 合成涂片图（已知 5 个红细胞 / 1 个白细胞 / 1 个血小板）验证传统计数 | `test_classic_cell_counter_on_synthetic_smear` |
| 跨后端一致性 | 同一图片对比 Python / C++ 输出 | 阶段 5 手工 + benchmark |
| 性能 | p50 / p95 / 吞吐 | `scripts/benchmark.py`、`bench_detector` |

---

## 10. 演示脚本（面试 5 分钟）

1. **（30 s）背景**：肠镜息肉漏检率高，AI 辅助可以实时提示。声明仅供技术演示。
2. **（1 min）图片检测**：上传内镜图 → 框出息肉 → 拖动阈值滑块 → 解释"为什么推荐阈值偏低：漏诊代价 > 误报"。翻到指标页看 Recall / 阈值扫描表。
3. **（1 min）视频实时**：播放内镜视频，看 HUD 的 FPS / 延迟；把发送宽度从 640 改到 320，FPS 上升。
4. **（1 min）后端切换**：切到 C++，对比延迟；讲 Net 池、线程安全、预处理对齐，以及下一步的 TensorRT。
5. **（1 min）血细胞对比**：传统分水岭 vs YOLO，讲各自的适用场景和失败模式。
6. **（30 s）工程化**：Docker 一键部署、模型即插即用（ONNX + JSON 契约）、测试不依赖训练结果。

---

## 11. 风险与对策

| 风险 | 对策 |
|---|---|
| CPU 机视频卡顿 | n 模型 + 降低发送分辨率 + INT8 量化；必要时只演示 GPU 机 |
| OpenCV DNN 不支持新算子 | 导出时用 opset 12；OpenCV ≥ 4.7；不行就在 C++ 改用 ONNX Runtime C++ API |
| 两个后端结果不一致 | letterbox 取整方式、RGB 顺序、归一化统一；用同一张图逐框对比 |
| 数据集许可 | Kvasir-SEG 仅限研究 / 教学：不商用，README 注明出处；不在仓库里分发整个数据集 |
| Ultralytics 是 AGPL-3.0 | 演示项目开源发布没问题；商用需要商业授权或换 Apache 许可的检测器（如 RT-DETR 官方实现） |
| 被误用于临床 | 页面顶部、报告、README 三处免责声明 |
| 公网暴露 | 上传大小限制（30 MB）、只读挂载 models、不保存用户上传的图片 |

---

## 12. 扩展方向

- 息肉**分割**（`prepare_kvasir.py --seg` + YOLO-seg），前端画掩膜轮廓，并计算息肉面积。
- 骨折检测（GRAZPEDWRI-DX，自带 YOLO 格式，CC BY 4.0）、胸片 14 类病灶（VinBigData，DICOM）作为新模型加入 `models/`，不用改代码。
- 浏览器端推理（ONNX Runtime Web），挂到 GitHub Pages，零服务器。
- Grad-CAM / EigenCAM 热力图，提升可解释性。
