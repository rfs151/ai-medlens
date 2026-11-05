"""Fetch one 3D brain-tumour MRI case for the C++/OpenGL volume viewer and convert it to .mlvol.gz.

Source: Medical Segmentation Decathlon, Task01_BrainTumour (BraTS 2016/2017 subset), CC BY-SA 4.0.
The task is published as one uncompressed 7.6 GB tar on AWS Open Data. Because a tar is not
compressed, HTTP range requests can read just the two members we need (~9 MB) instead of the archive.

    python scripts/fetch_volume.py            # -> samples/volume/brats_206.mlvol.gz

.mlvol layout (little endian), read by gl_viewer/src/volume.cc:
    char     magic[8]   "MLVOL01\\0"
    int32    nx, ny, nz             voxels; x varies fastest (matches glTexImage3D width/height/depth)
    float32  sx, sy, sz             voxel spacing in mm
    int32    channels               number of 8-bit intensity volumes that follow
    int32    has_labels             1 if a label volume follows the channels
    char     names[4][16]           channel names (ASCII)
    ...      zero padding to 128 bytes
    uint8    channel data, nx*ny*nz each (0 = background, 1..255 = brain)
    uint8    labels, nx*ny*nz       0 background, 1 edema, 2 non-enhancing tumour, 3 enhancing tumour
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "samples" / "volume"
URL = "https://msd-for-monai.s3.amazonaws.com/Task01_BrainTumour.tar"
CASE = "BRATS_206"
# (offset of the member's data inside the tar, size, sha256), located once by walking the tar headers
MEMBERS = {
    "image": (21630976 + 512, 8875547, "cb7547bb3ea75ca639409e71c71ffcaaad2de2a61f7ac4bc0b304636d7f12c06"),
    "label": (7593720832 + 512, 51943, "2be4fb1bf53e600f00240e7c35bc4f8617861a7627cb91c116bbc25bb9ed9890"),
}
# MSD channel order is FLAIR, T1w, T1gd, T2w; keep the three that show the tumour best
CHANNELS = [(0, "FLAIR"), (2, "T1Gd"), (3, "T2")]
MARGIN = 3


def fetch(offset: int, size: int, sha: str, dest: Path) -> bytes:
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == sha:
        return dest.read_bytes()
    print(f"  downloading {size / 1e6:.1f} MB from byte {offset} ...", flush=True)
    # curl handles proxies (HTTPS_PROXY) and retries far better than urllib on slow links
    subprocess.run(["curl", "-sSL", "--retry", "5", "--retry-all-errors", "-r", f"{offset}-{offset + size - 1}",
                    "-o", str(dest), URL], check=True)
    data = dest.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        dest.unlink()
        sys.exit(f"checksum mismatch for {dest.name}")
    return data


def read_nifti(gz: bytes) -> tuple[np.ndarray, tuple[float, float, float]]:
    raw = gzip.decompress(gz)
    dim = struct.unpack("<8h", raw[40:56])
    datatype = struct.unpack("<h", raw[70:72])[0]
    pixdim = struct.unpack("<8f", raw[76:108])
    vox_offset = int(struct.unpack("<f", raw[108:112])[0])
    dtype = {2: np.uint8, 4: np.int16, 16: np.float32}[datatype]
    shape = tuple(d for d in dim[1:dim[0] + 1])
    # NIfTI stores x fastest (Fortran order); reverse to (…, z, y, x) in C order
    arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder("<"), offset=vox_offset,
                        count=int(np.prod(shape))).reshape(shape[::-1])
    return arr, pixdim[1:4]


def to_uint8(vol: np.ndarray, brain: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(vol[brain], [0.5, 99.5])
    out = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1) * 254 + 1
    return np.where(brain, out, 0).astype(np.uint8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = OUT_DIR / ".cache"
    cache.mkdir(exist_ok=True)
    print(f"{CASE} (MSD Task01_BrainTumour)")
    img_gz = fetch(*MEMBERS["image"], cache / f"{CASE}_image.nii.gz")
    lab_gz = fetch(*MEMBERS["label"], cache / f"{CASE}_label.nii.gz")
    img, spacing = read_nifti(img_gz)  # (4, z, y, x)
    lab, _ = read_nifti(lab_gz)        # (z, y, x)

    brain = img[0] > 0
    idx = np.argwhere(brain)
    lo = np.maximum(idx.min(0) - MARGIN, 0)
    hi = np.minimum(idx.max(0) + MARGIN + 1, brain.shape)
    crop = tuple(slice(a, b) for a, b in zip(lo, hi))
    brain = brain[crop]
    nz, ny, nx = brain.shape

    header = b"MLVOL01\0" + struct.pack("<3i3f2i", nx, ny, nz, *spacing, len(CHANNELS), 1)
    header += b"".join(name.encode().ljust(16, b"\0") for _, name in CHANNELS)
    header += b"\0" * (4 * 16 - 16 * len(CHANNELS))
    header = header.ljust(128, b"\0")
    body = b"".join(to_uint8(img[c][crop], brain).tobytes() for c, _ in CHANNELS)
    body += lab[crop].astype(np.uint8).tobytes()

    out = OUT_DIR / f"{CASE.lower()}.mlvol.gz"
    out.write_bytes(gzip.compress(header + body, compresslevel=9, mtime=0))
    voxel_ml = float(np.prod(spacing)) / 1000.0
    counts = np.bincount(lab[crop].ravel(), minlength=4)
    print(f"  -> {out.relative_to(ROOT)}  {nx}x{ny}x{nz}  {out.stat().st_size / 1e6:.1f} MB")
    print("  tumour volume (ml): edema %.1f, non-enhancing %.1f, enhancing %.1f"
          % tuple(counts[1:4] * voxel_ml))

    manifest_path = ROOT / "samples" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["volume3d"] = {"volumes": [{
        "file": f"volume/{out.name}",
        "caption": f"脑胶质瘤 MRI · {CASE}",
        "dims": [nx, ny, nz],
        "channels": [name for _, name in CHANNELS],
        "source": "Medical Segmentation Decathlon Task01_BrainTumour (BraTS)",
        "url": "http://medicaldecathlon.com/",
        "license": "CC BY-SA 4.0",
    }]}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
