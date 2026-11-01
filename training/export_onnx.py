"""Export trained weights to ONNX and drop them into models/ with a sidecar JSON.

The sidecar (<name>.json) carries class names, input size, description and license,
so both the Python (ONNX Runtime) and C++ (OpenCV DNN) backends read the same metadata.

Usage:
    python training/export_onnx.py --weights runs/polyp/weights/best.pt --name polyp \
        --description "肠镜息肉检测 (Kvasir-SEG)" --license "Kvasir-SEG: research & educational use only"

Smoke-test the whole pipeline without training (COCO pretrained):
    python training/export_onnx.py --weights yolo11n.pt --name coco-smoke
"""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=12, help="12 works with OpenCV DNN >= 4.7")
    ap.add_argument("--half", action="store_true", help="FP16 (GPU only)")
    ap.add_argument("--description", default="")
    ap.add_argument("--license", default="")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = ap.parse_args()

    model = YOLO(args.weights)
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True,
                                  dynamic=False, half=args.half))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dst = args.out_dir / f"{args.name}.onnx"
    shutil.copy2(onnx_path, dst)

    sidecar = dst.with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    names = model.names
    meta.update({
        "names": [names[i] for i in sorted(names)],
        "imgsz": args.imgsz,
        "task": model.task,
        "source_weights": str(args.weights),
    })
    if args.description:
        meta["description"] = args.description
    if args.license:
        meta["license"] = args.license
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported {dst} (+ {sidecar.name}), classes={meta['names']}")


if __name__ == "__main__":
    main()
