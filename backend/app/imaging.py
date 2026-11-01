"""Image decoding: JPG/PNG via OpenCV, DICOM via pydicom (with window width / level)."""
from __future__ import annotations

import io

import cv2
import numpy as np


class ImageDecodeError(ValueError):
    pass


def is_dicom(data: bytes, filename: str = "") -> bool:
    return filename.lower().endswith((".dcm", ".dicom")) or (len(data) > 132 and data[128:132] == b"DICM")


def apply_window(pixels: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map raw values to 0..255 using window center / width (窗位 / 窗宽)."""
    lo, hi = center - width / 2.0, center + width / 2.0
    out = (pixels.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def decode_dicom(data: bytes, wc: float | None = None, ww: float | None = None) -> tuple[np.ndarray, dict]:
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover
        raise ImageDecodeError("pydicom is not installed") from e

    ds = pydicom.dcmread(io.BytesIO(data), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):  # multi-frame: take the first frame
        arr = arr[0]
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    arr = arr * slope + intercept

    def _first(v):
        if v is None:
            return None
        return float(v[0]) if isinstance(v, pydicom.multival.MultiValue) else float(v)

    wc = wc if wc is not None else _first(getattr(ds, "WindowCenter", None))
    ww = ww if ww is not None else _first(getattr(ds, "WindowWidth", None))
    if wc is None or ww is None:  # fall back to full dynamic range
        lo, hi = float(arr.min()), float(arr.max())
        wc, ww = (lo + hi) / 2, max(hi - lo, 1.0)

    img = apply_window(arr, wc, ww)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = 255 - img
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    info = {
        "modality": str(getattr(ds, "Modality", "")),
        "window_center": wc,
        "window_width": ww,
        "rows": int(getattr(ds, "Rows", img.shape[0])),
        "columns": int(getattr(ds, "Columns", img.shape[1])),
    }
    return img, info


def decode_image(data: bytes, filename: str = "", wc: float | None = None, ww: float | None = None):
    """Returns (BGR image, dicom_info or None)."""
    if not data:
        raise ImageDecodeError("empty file")
    if is_dicom(data, filename):
        return decode_dicom(data, wc, ww)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ImageDecodeError("unsupported image format")
    return img, None


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()
