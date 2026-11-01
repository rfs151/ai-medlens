"""Download demo images / videos for each imaging menu into samples/ and write samples/manifest.json.

    python scripts/fetch_samples.py            # needs curl + ffmpeg (with libx264)
    HF_ENDPOINT=https://hf-mirror.com python scripts/fetch_samples.py

All sources are public datasets with licenses that allow redistribution with attribution
(see the "license" field of each entry; the page shows it under every sample).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples"
HF = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

HYPERKVASIR = ("HyperKvasir polyp videos (Borgli et al., Scientific Data 2020) · "
               "huggingface.co/datasets/borgli/HyperKvasir-Polyp-Videos", "CC BY 4.0 / MIT")
BCCD = ("BCCD Dataset · github.com/Shenggan/BCCD_Dataset", "MIT")
FRACTURE = ("Roboflow-100 bone-fracture-7fylg · huggingface.co/datasets/LibreYOLO/bone-fracture-7fylg", "CC BY 4.0")
BRAIN = ("Brain Tumor MRI Dataset (M. Nickparvar; figshare / SARTAJ / Br35H) · "
         "huggingface.co/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset", "CC BY 4.0")
DENTEX = ("DENTEX Challenge 2023 (Hamamci et al.) · huggingface.co/datasets/tirandazdylan/dentex-panoramic", "CC BY 4.0")

POLYP_VIDEOS = [  # (file in the HF dataset, start second, caption, seconds of the still frames)
    # still frames picked where the polyp model gives a confident, localized box
    ("5b2f37fb-8365-4b75-85c1-731e5ce120e6.mp4", 0, "结肠镜检查片段 1", (6.2, 22.8)),
    ("aaba39c3-8037-4178-82c8-b89a657f981c.mp4", 0, "结肠镜检查片段 2", (5.4, 21.2)),
]
BCCD_IMAGES = ["BloodImage_00000.jpg", "BloodImage_00010.jpg", "BloodImage_00023.jpg"]
FRACTURE_IMAGES = [
    "test/images/28_jpg.rf.24ca36c7dbf8209c85f518b4e8e1540c.jpg",
    "test/images/164_jpg.rf.8fe7e5ddd7b47eb6aa45e43b9879101c.jpg",
    "test/images/194_jpg.rf.d73379754050a514bb51098c06f1a110.jpg",
]
BRAIN_IMAGES = [
    ("Testing/glioma/Te-gl_14.jpg", "胶质瘤 (glioma)"),
    ("Testing/meningioma/Te-me_10.jpg", "脑膜瘤 (meningioma)"),
    ("Testing/pituitary/Te-pi_11.jpg", "垂体瘤 (pituitary)"),
    ("Testing/notumor/Te-no_10.jpg", "无肿瘤 (对照)"),
]
DENTEX_ROWS = 3


def curl(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {url}")
    subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(dst), url], check=True)


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def entry(file: Path, caption: str, src: tuple[str, str]) -> dict:
    return {"file": file.relative_to(OUT).as_posix(), "caption": caption, "source": src[0], "license": src[1]}


def polyp(tmp: Path) -> dict:
    images, videos = [], []
    for i, (name, start, caption, stills) in enumerate(POLYP_VIDEOS, 1):
        raw = tmp / name
        curl(f"{HF}/datasets/borgli/HyperKvasir-Polyp-Videos/resolve/main/{name}", raw)
        clip = OUT / "polyp" / f"colonoscopy_{i}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        # browser friendly: H.264 / yuv420p / 640 px wide / no audio / moov atom first
        ffmpeg("-ss", str(start), "-i", str(raw), "-t", "30", "-vf", "scale=640:-2", "-an",
               "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(clip))
        videos.append(entry(clip, caption, HYPERKVASIR))
        for j, t in enumerate(stills, 1):  # still frames from the clip as image samples
            img = OUT / "polyp" / f"colonoscopy_{i}_frame{j}.jpg"
            ffmpeg("-ss", str(t), "-i", str(clip), "-frames:v", "1", "-q:v", "3", str(img))
            images.append(entry(img, f"{caption} · 第 {t} 秒截帧", HYPERKVASIR))
    return {"images": images, "videos": videos}


def blood_cell(tmp: Path) -> dict:
    images = []
    for name in BCCD_IMAGES:
        dst = OUT / "blood-cell" / name.lower()
        curl(f"https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/{name}", dst)
        images.append(entry(dst, f"血涂片 {name.split('_')[1][:5]}", BCCD))
    video = OUT / "blood-cell" / "microscope_pan.mp4"
    make_pan_video([OUT / "blood-cell" / n.lower() for n in BCCD_IMAGES], video)
    return {"images": images, "videos": [entry(video, "模拟显微镜视野平移（由 3 张血涂片拼接生成）", BCCD)]}


def make_pan_video(paths: list[Path], dst: Path, fps: int = 25, seconds: int = 24) -> None:
    """Stitch smears side by side and slide a 480x360 window across: imitates moving the slide."""
    tiles = [cv2.imread(str(p)) for p in paths]
    h = min(t.shape[0] for t in tiles)
    strip = np.hstack([t[:h] for t in tiles])
    win_w, win_h = 480, 360
    n = fps * seconds
    with tempfile.TemporaryDirectory() as d:
        for i in range(n):
            phase = i / (n - 1)
            x = int((strip.shape[1] - win_w) * (0.5 - 0.5 * np.cos(np.pi * phase)))  # ease in/out
            y = int((h - win_h) * (0.5 + 0.5 * np.sin(2 * np.pi * phase)))
            frame = cv2.resize(strip[y:y + win_h, x:x + win_w], (640, 480), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{d}/{i:05d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ffmpeg("-framerate", str(fps), "-i", f"{d}/%05d.jpg", "-c:v", "libx264", "-preset", "slow", "-crf", "24",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst))


def fracture(_: Path) -> dict:
    images = []
    for i, path in enumerate(FRACTURE_IMAGES, 1):
        dst = OUT / "fracture" / f"xray_{i}.jpg"
        curl(f"{HF}/datasets/LibreYOLO/bone-fracture-7fylg/resolve/main/{path}", dst)
        images.append(entry(dst, f"骨骼 X 光 {i}", FRACTURE))
    return {"images": images, "videos": []}


def brain_tumor(_: Path) -> dict:
    images = []
    for path, caption in BRAIN_IMAGES:
        dst = OUT / "brain-tumor" / Path(path).name.lower()
        curl(f"{HF}/datasets/BJyotibrat/Masoud-Nickparvar-Brain-Tumor-MRI-Dataset/resolve/main/{path}", dst)
        images.append(entry(dst, f"T1 增强 MRI · {caption}", BRAIN))
    return {"images": images, "videos": []}


def dental(tmp: Path) -> dict:
    api = ("https://datasets-server.huggingface.co/rows?dataset=tirandazdylan/dentex-panoramic"
           f"&config=diagnosis&split=validation&offset=0&length={DENTEX_ROWS}")
    with urllib.request.urlopen(api, timeout=60) as r:
        rows = json.load(r)["rows"]
    images = []
    for i, r in enumerate(rows, 1):
        raw = tmp / f"dentex_{i}.img"
        curl(r["row"]["image"]["src"], raw)
        img = cv2.imread(str(raw))
        scale = 1600 / img.shape[1]
        img = cv2.resize(img, (1600, round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        dst = OUT / "dental" / f"panoramic_{i}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        images.append(entry(dst, f"口腔全景片 {i}（{r['row']['image_id']}）", DENTEX))
    return {"images": images, "videos": []}


STEPS = {"polyp": polyp, "blood-cell": blood_cell, "fracture": fracture, "brain-tumor": brain_tumor, "dental": dental}


def main():
    for tool in ("curl", "ffmpeg"):
        if not shutil.which(tool):
            sys.exit(f"{tool} not found")
    wanted = sys.argv[1:] or list(STEPS)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        for key in wanted:
            print(f"[{key}]")
            manifest[key] = STEPS[key](Path(tmp))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
