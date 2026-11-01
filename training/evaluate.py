"""Validate a model and write metrics into models/<name>.json for the web "模型指标" tab.

Also sweeps the confidence threshold and picks a recall-oriented recommended threshold:
the highest conf whose recall >= --target-recall (missing a lesion costs more than a false alarm).

Usage:
    python training/evaluate.py --weights runs/polyp/weights/best.pt --data training/data/polyp.yaml --name polyp
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="0")
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--confs", default="0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    args = ap.parse_args()

    model = YOLO(args.weights)
    base = model.val(data=args.data, imgsz=args.imgsz, split=args.split, device=args.device, plots=True,
                     project="runs", name=f"{args.name}-val", exist_ok=True)
    metrics = {
        "split": args.split,
        "map50": float(base.box.map50),
        "map50_95": float(base.box.map),
        "precision": float(base.box.mp),
        "recall": float(base.box.mr),
        "per_class_map50_95": {model.names[int(c)]: float(m) for c, m in zip(base.box.ap_class_index, base.box.maps[base.box.ap_class_index])},
    }

    sweep = []
    for conf in (float(c) for c in args.confs.split(",")):
        r = model.val(data=args.data, imgsz=args.imgsz, split=args.split, device=args.device,
                      conf=conf, plots=False, verbose=False, project="runs", name=f"{args.name}-sweep", exist_ok=True)
        sweep.append({"conf": conf, "precision": float(r.box.mp), "recall": float(r.box.mr), "map50": float(r.box.map50)})
        print(f"conf={conf:.2f}  P={r.box.mp:.3f}  R={r.box.mr:.3f}")
    metrics["threshold_sweep"] = sweep

    ok = [s for s in sweep if s["recall"] >= args.target_recall]
    recommended = max(ok, key=lambda s: s["conf"])["conf"] if ok else min(s["conf"] for s in sweep)

    sidecar = ROOT / "models" / f"{args.name}.json"
    meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    meta["metrics"] = metrics
    meta["recommended_conf"] = recommended
    meta["target_recall"] = args.target_recall
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"recommended conf (recall >= {args.target_recall}): {recommended} -> {sidecar}")


if __name__ == "__main__":
    main()
