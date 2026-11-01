"""Train a YOLO detector with Ultralytics.

Examples:
    python training/train.py --data training/data/polyp.yaml --name polyp --model yolo11s.pt --epochs 100
    python training/train.py --data training/data/bccd.yaml  --name bccd  --model yolo11n.pt --epochs 80
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", required=True, help="run name, also used as model name later")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n/s/m.pt, yolov8n.pt ...")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto batch for GPU memory")
    ap.add_argument("--device", default="0", help="'0', '0,1' or 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project="runs",
        name=args.name,
        exist_ok=True,
        # medical-image friendly augmentation: endoscopy/x-ray have no canonical orientation
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.01,  # keep colour shifts small: colour carries diagnostic information
        plots=True,
    )
    print(f"best weights: runs/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
