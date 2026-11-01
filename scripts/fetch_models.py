"""Download ready-made third-party medical YOLO models into models/ (no training needed).

Each model ends up as models/<name>.onnx + models/<name>.json (class names, description,
source, license, model-card metrics, recommended thresholds).

    python scripts/fetch_models.py                 # all models
    python scripts/fetch_models.py polyp blood-cell
    HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_models.py   # mainland China mirror

Models published only as PyTorch .pt are exported to ONNX with Ultralytics, which needs
`pip install ultralytics` (pulls in PyTorch). Models published as .onnx need nothing extra.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
HF = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# metrics are copied from each model card (validation split of the author's dataset)
ZOO: dict[str, dict] = {
    "polyp": {
        # picked over sdoerrich97/true_colon_yolov11m_realcolon_s42 (REAL-Colon): on the HyperKvasir demo videos
        # that model fires on ~1% of frames vs ~50% here (Kvasir-SEG comes from the same hospital / equipment)
        "repo": "pourmand1376/Kvasir-SEG-yolov8",
        "file": "kvasir-yolov8-best.pt",
        "names": ["polyp"],  # checkpoint labels its single class "json"
        "description": "结肠镜息肉检测 · YOLOv8 · Kvasir-SEG 1000 张息肉图训练（与示例视频同源：挪威 Bærum 医院）",
        "license": "MIT（模型权重）· 训练数据 Kvasir-SEG 仅限研究与教学",
        "recommended_conf": 0.30,
        "max_box_area": 0.4,  # drop the occasional box covering the whole frame (~8% of frames otherwise)
    },
    "blood-cell": {
        "repo": "Jesteban247/yolo11-blood_cell-onnx",
        "file": "best.onnx",
        "description": "血细胞检测 · YOLO11n · 红细胞、血小板与 5 类白细胞（Roboflow Blood Cell 数据集）",
        "license": "Apache-2.0",
        "metrics": {"map50": 0.974, "map50_95": 0.905, "precision": 0.951, "recall": 0.920, "source": "model card"},
        "count_groups": {"Basophil": "WBC", "Eosinophil": "WBC", "Lymphocyte": "WBC", "Monocyte": "WBC",
                         "Neutrophil": "WBC", "Platelets": "Platelets", "RBC": "RBC"},
        "recommended_conf": 0.25,
    },
    "fracture": {
        "repo": "Jesteban247/yolo11-fracture-onnx",
        "file": "best.onnx",
        "description": "X 光骨折检测 · YOLO11n（Roboflow Fracture 数据集）",
        "license": "MIT",
        "metrics": {"map50": 0.920, "map50_95": 0.524, "precision": 0.903, "recall": 0.832, "source": "model card"},
        "recommended_conf": 0.20,
    },
    "brain-tumor": {
        "repo": "amcm/brain-tumor-yolo11n",
        "file": "brain_tumor_yolo11n_ft.pt",
        "description": "脑 MRI 肿瘤检测 · YOLO11n · 胶质瘤 / 脑膜瘤 / 垂体瘤（Kaggle Brain Tumor YOLO 数据集）",
        "license": "AGPL-3.0（Ultralytics 基座）· 数据集 Apache-2.0",
        "metrics": {"map50": 0.961, "map50_95": 0.667, "precision": 0.883, "recall": 0.932, "source": "model card"},
        # measured on 27 tumor + 9 normal MRIs from the demo dataset: 0.25 -> 5/9 false alarms, 0.35 -> 3/9
        "recommended_conf": 0.35,
    },
    "dental": {
        "repo": "liodon-ai/dental-panoramic-detector",
        "file": "best.onnx",
        "description": "口腔全景片检测 · YOLO11n · 龋齿 / 根尖周病变 / 阻生齿（DENTEX + OralXrays-9）",
        "license": "CC BY-NC 4.0（仅限非商业用途）",
        "metrics": {"map50": 0.622, "map50_95": 0.406, "precision": 0.630, "recall": 0.614, "source": "model card"},
        "recommended_conf": 0.45,  # per model card: lower values over-fire on adjacent teeth
        "recommended_iou": 0.35,
    },
}


def download(url: str, dst: Path) -> None:
    print(f"  downloading {url}")
    if shutil.which("curl"):  # much faster than urllib behind some HTTP proxies
        subprocess.run(["curl", "-fL", "--retry", "5", "--retry-all-errors", "-#", "-o", str(dst), url], check=True)
        return
    req = urllib.request.Request(url, headers={"User-Agent": "medlens-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def onnx_names(path: Path) -> list[str] | None:
    """Class names from Ultralytics ONNX metadata, e.g. "{0: 'RBC', 1: 'WBC'}"."""
    import ast

    import onnx

    meta = {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}
    if "names" not in meta:
        return None
    names = ast.literal_eval(meta["names"])
    return [names[k] for k in sorted(names)]


def export_pt(pt: Path, out: Path, imgsz: int) -> list[str]:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("this model is published as .pt: `pip install ultralytics` first to export it to ONNX")
    model = YOLO(str(pt))
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False))
    shutil.move(onnx_path, out)
    return [model.names[i] for i in sorted(model.names)]


def fetch(name: str, spec: dict, force: bool) -> None:
    dst = MODELS_DIR / f"{name}.onnx"
    print(f"[{name}] {spec['repo']}")
    if dst.exists() and not force:
        print("  already present, skipping download (use --force to refresh)")
        names = onnx_names(dst)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / spec["file"]
            download(f"{HF}/{spec['repo']}/resolve/main/{spec['file']}", src)
            if src.suffix == ".pt":
                names = export_pt(src, dst, spec.get("imgsz", 640))
            else:
                shutil.copy2(src, dst)
                names = onnx_names(dst)

    meta = {
        "names": spec.get("names") or names,
        "imgsz": spec.get("imgsz", 640),
        "description": spec["description"],
        "license": spec["license"],
        "source": f"https://huggingface.co/{spec['repo']}",
    }
    for key in ("metrics", "recommended_conf", "recommended_iou", "max_box_area", "count_groups"):
        if key in spec:
            meta[key] = spec[key]
    dst.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> {dst.relative_to(ROOT)} classes={meta['names']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", help=f"subset of: {', '.join(ZOO)}")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()
    unknown = set(args.models) - set(ZOO)
    if unknown:
        sys.exit(f"unknown model(s): {', '.join(unknown)}; choose from {', '.join(ZOO)}")
    MODELS_DIR.mkdir(exist_ok=True)
    for name in args.models or ZOO:
        fetch(name, ZOO[name], args.force)


if __name__ == "__main__":
    main()
