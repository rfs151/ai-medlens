"""Draw detection boxes onto an image (used by /api/detect/render and reports)."""
from __future__ import annotations

import cv2
import numpy as np

PALETTE = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72)]


def draw_boxes(img: np.ndarray, boxes: list[dict], show_conf: bool = True) -> np.ndarray:
    out = img.copy()
    thick = max(1, round(sum(out.shape[:2]) / 600))
    for b in boxes:
        color = PALETTE[b["cls"] % len(PALETTE)]
        p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(out, p1, p2, color, thick, cv2.LINE_AA)
        text = f'{b["label"]} {b["conf"]:.2f}' if show_conf else b["label"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(out, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), color, -1)
        cv2.putText(out, text, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
