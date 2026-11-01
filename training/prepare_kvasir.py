"""Kvasir-SEG (images + binary masks) -> YOLO detection dataset.

Download: https://datasets.simula.no/kvasir-seg/  (Kvasir-SEG.zip, ~46 MB)
Expected layout after unzip:
    <src>/images/*.jpg
    <src>/masks/*.jpg

Each connected component of the mask becomes one 'polyp' box.
With --seg, polygon labels are written instead (for YOLO-seg training).

Usage:
    python training/prepare_kvasir.py --src datasets/raw/Kvasir-SEG --dst datasets/polyp
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def mask_to_labels(mask: np.ndarray, min_area_ratio: float, seg: bool) -> list[str]:
    h, w = mask.shape
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for c in contours:
        if cv2.contourArea(c) < min_area_ratio * h * w:
            continue
        if seg:
            c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(c) < 3:
                continue
            pts = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in c)
            lines.append(f"0 {pts}")
        else:
            x, y, bw, bh = cv2.boundingRect(c)
            lines.append(f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--min-area", type=float, default=0.001, help="ignore components smaller than this image fraction")
    ap.add_argument("--seg", action="store_true", help="write polygon labels for YOLO-seg")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    images = sorted((args.src / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images under {args.src / 'images'}")
    random.Random(args.seed).shuffle(images)
    n_val = int(len(images) * args.val_ratio)
    splits = {"val": images[:n_val], "train": images[n_val:]}

    stats = {"images": 0, "boxes": 0, "empty": 0}
    for split, files in splits.items():
        (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img_path in files:
            mask = cv2.imread(str(args.src / "masks" / img_path.name), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"skip {img_path.name}: no mask")
                continue
            lines = mask_to_labels(mask, args.min_area, args.seg)
            shutil.copy2(img_path, args.dst / "images" / split / img_path.name)
            (args.dst / "labels" / split / f"{img_path.stem}.txt").write_text("\n".join(lines))
            stats["images"] += 1
            stats["boxes"] += len(lines)
            stats["empty"] += not lines
    print(f"done: {stats}, train={len(splits['train'])}, val={len(splits['val'])} -> {args.dst}")


if __name__ == "__main__":
    main()
