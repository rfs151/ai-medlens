"""Traditional blood-cell counting (no deep learning), used to compare against YOLO.

Pipeline:
  WBC / platelets : stained purple -> high saturation in HSV -> threshold -> contours,
                    split by area (big = WBC, small = platelet)
  RBC             : Otsu on the green channel -> remove purple regions -> morphology
                    -> distance transform -> watershed to split touching cells
"""
from __future__ import annotations

import time

import cv2
import numpy as np


def _boxes_from_contours(contours, label: str, cls: int, min_area: float, max_area: float):
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(c)
            out.append({"cls": cls, "label": label, "conf": 1.0, "x1": x, "y1": y, "x2": x + w, "y2": y + h})
    return out


def count_cells(img_bgr: np.ndarray, params: dict | None = None) -> dict:
    p = {
        "sat_thresh": 90,  # purple stain saturation threshold
        "wbc_min_area_ratio": 0.004,  # relative to image area
        "plt_min_area_ratio": 0.00005,
        "rbc_min_area_ratio": 0.0008,
        "rbc_max_area_ratio": 0.03,
        "dist_ratio": 0.45,  # watershed seed threshold
    }
    p.update(params or {})
    t0 = time.perf_counter()
    h, w = img_bgr.shape[:2]
    area = float(h * w)
    blur = cv2.GaussianBlur(img_bgr, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # --- WBC + platelets: strongly stained (purple) regions ---
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(hsv, (110, p["sat_thresh"], 30), (170, 255, 255))
    purple = cv2.morphologyEx(purple, cv2.MORPH_OPEN, kernel)
    purple_closed = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(purple_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wbc_min = p["wbc_min_area_ratio"] * area
    wbc = _boxes_from_contours(contours, "WBC", 1, wbc_min, area)
    plt = _boxes_from_contours(contours, "Platelets", 2, p["plt_min_area_ratio"] * area, wbc_min - 1)

    # --- RBC: cell foreground on green channel, minus purple regions, then watershed ---
    green = blur[:, :, 1]
    _, fg = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    fg[cv2.dilate(purple_closed, kernel, iterations=2) > 0] = 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=2)
    fg = _fill_holes(fg)  # RBCs have pale centers

    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)
    _, seeds = cv2.threshold(dist, p["dist_ratio"] * dist.max() if dist.max() > 0 else 1, 255, cv2.THRESH_BINARY)
    seeds = seeds.astype(np.uint8)
    n_seeds, markers = cv2.connectedComponents(seeds)
    markers = markers + 1
    unknown = cv2.subtract(cv2.dilate(fg, kernel, iterations=3), seeds)
    markers[unknown == 255] = 0
    markers = cv2.watershed(blur.copy(), markers)

    rbc = []
    rmin, rmax = p["rbc_min_area_ratio"] * area, p["rbc_max_area_ratio"] * area
    for lbl in range(2, n_seeds + 1):
        mask = (markers == lbl).astype(np.uint8)
        a = float(mask.sum())
        if rmin <= a <= rmax:
            x, y, bw, bh = cv2.boundingRect(mask)
            rbc.append({"cls": 0, "label": "RBC", "conf": 1.0, "x1": x, "y1": y, "x2": x + bw, "y2": y + bh})

    boxes = rbc + wbc + plt
    return {
        "method": "opencv-threshold-watershed",
        "image": {"width": w, "height": h},
        "boxes": boxes,
        "counts": {"RBC": len(rbc), "WBC": len(wbc), "Platelets": len(plt)},
        "timing_ms": {"total": round((time.perf_counter() - t0) * 1000, 2)},
    }


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    # pad so the flood fill seed (0, 0) is always background
    flood = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h, w = flood.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return mask | holes
