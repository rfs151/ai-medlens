"""BCCD (Pascal VOC XML) -> YOLO detection dataset.

Download: git clone https://github.com/Shenggan/BCCD_Dataset   (MIT license)
Expected layout:
    <src>/BCCD/Annotations/*.xml
    <src>/BCCD/JPEGImages/*.jpg
    <src>/BCCD/ImageSets/Main/{train,val,test}.txt

Usage:
    python training/prepare_bccd.py --src datasets/raw/BCCD_Dataset --dst datasets/bccd
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["RBC", "WBC", "Platelets"]


def convert(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    w = float(root.findtext("size/width"))
    h = float(root.findtext("size/height"))
    lines = []
    for obj in root.iter("object"):
        name = obj.findtext("name").strip()
        if name not in CLASSES:
            continue
        bb = obj.find("bndbox")
        x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
        x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        x1, x2 = max(0.0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(f"{CLASSES.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dst", required=True, type=Path)
    args = ap.parse_args()

    base = args.src / "BCCD"
    sets = base / "ImageSets" / "Main"
    for split in ("train", "val", "test"):
        ids = [s.strip() for s in (sets / f"{split}.txt").read_text().split() if s.strip()]
        (args.dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_boxes = 0
        for i in ids:
            img = base / "JPEGImages" / f"{i}.jpg"
            xml = base / "Annotations" / f"{i}.xml"
            if not img.exists() or not xml.exists():
                continue
            lines = convert(xml)
            n_boxes += len(lines)
            shutil.copy2(img, args.dst / "images" / split / img.name)
            (args.dst / "labels" / split / f"{i}.txt").write_text("\n".join(lines))
        print(f"{split}: {len(ids)} images, {n_boxes} boxes")


if __name__ == "__main__":
    main()
