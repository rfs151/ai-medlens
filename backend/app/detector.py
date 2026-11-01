"""YOLO detector on ONNX Runtime: letterbox -> inference -> decode -> NMS.

Handles both output layouts:
  * YOLOv8 / YOLO11 : (1, 4 + nc, N)      boxes are cx, cy, w, h + class scores
  * YOLOv5          : (1, N, 5 + nc)      boxes are cx, cy, w, h, objectness + class scores
"""
from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Box:
    cls: int
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self) -> dict:
        return {
            "cls": self.cls,
            "label": self.label,
            "conf": round(self.conf, 4),
            "x1": round(self.x1, 1),
            "y1": round(self.y1, 1),
            "x2": round(self.x2, 1),
            "y2": round(self.y2, 1),
        }


@dataclass
class LetterboxInfo:
    scale: float
    pad_x: float
    pad_y: float


@dataclass
class DetectResult:
    boxes: list[Box]
    width: int
    height: int
    timing_ms: dict = field(default_factory=dict)


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio, pad to size x size (same as Ultralytics)."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (size - nw) / 2, (size - nh) / 2
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_y - 0.1)), int(round(pad_y + 0.1))
    left, right = int(round(pad_x - 0.1)), int(round(pad_x + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, LetterboxInfo(scale, left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Plain NMS. boxes: (N, 4) xyxy. Returns kept indices sorted by score."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return np.array(keep, dtype=np.int64)


def decode(output: np.ndarray, num_classes: int, conf_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw model output -> (xyxy boxes, scores, class ids) in letterboxed coordinates."""
    out = np.squeeze(output, 0)
    if out.shape[0] == 4 + num_classes and out.shape[1] != 4 + num_classes:
        # YOLOv8/11 layout (4+nc, N) -> (N, 4+nc)
        out = out.T
        cls_scores = out[:, 4:]
    elif out.shape[1] == 5 + num_classes:
        # YOLOv5 layout: objectness * class prob
        cls_scores = out[:, 5:] * out[:, 4:5]
    elif out.shape[1] == 4 + num_classes:
        cls_scores = out[:, 4:]
    else:
        raise ValueError(f"Unexpected output shape {output.shape} for {num_classes} classes")

    class_ids = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls_scores)), class_ids]
    mask = scores >= conf_thr
    cxcywh, scores, class_ids = out[mask, :4], scores[mask], class_ids[mask]
    xyxy = np.empty_like(cxcywh)
    xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
    xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
    xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
    xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
    return xyxy, scores, class_ids


def batched_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_thr: float) -> np.ndarray:
    """Per-class NMS using the coordinate-offset trick."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    offset = class_ids[:, None].astype(boxes.dtype) * (boxes.max() + 1)
    return nms(boxes + offset, scores, iou_thr)


def load_meta(onnx_path: Path, session=None) -> dict:
    """Class names / input size: sidecar <name>.json first, then Ultralytics ONNX metadata."""
    meta: dict = {}
    sidecar = onnx_path.with_suffix(".json")
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if session is not None and "names" not in meta:
        custom = session.get_modelmeta().custom_metadata_map
        if "names" in custom:
            names = ast.literal_eval(custom["names"])  # "{0: 'polyp'}"
            meta["names"] = [names[k] for k in sorted(names)]
        if "imgsz" in custom and "imgsz" not in meta:
            meta["imgsz"] = ast.literal_eval(custom["imgsz"])[0]
    if isinstance(meta.get("names"), dict):
        meta["names"] = [meta["names"][k] for k in sorted(meta["names"], key=int)]
    return meta


class YoloOnnxDetector:
    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None, num_threads: int = 0):
        import onnxruntime as ort

        self.path = Path(onnx_path)
        self.name = self.path.stem
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            opts.intra_op_num_threads = num_threads
        available = ort.get_available_providers()
        wanted = providers or ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            str(self.path), sess_options=opts, providers=[p for p in wanted if p in available]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_fp16 = self.session.get_inputs()[0].type == "tensor(float16)"  # half=True exports
        self.meta = load_meta(self.path, self.session)

        shape = self.session.get_inputs()[0].shape  # [1, 3, H, W], may be symbolic
        self.imgsz = shape[2] if isinstance(shape[2], int) else int(self.meta.get("imgsz", 640))
        self.names: list[str] = list(self.meta.get("names") or [])
        if not self.names:
            out_shape = self.session.get_outputs()[0].shape
            nc = out_shape[1] - 4 if isinstance(out_shape[1], int) and out_shape[1] < 100 else 1
            self.names = [f"class{i}" for i in range(nc)]
        self.provider = self.session.get_providers()[0]

    def info(self) -> dict:
        return {
            "name": self.name,
            "classes": self.names,
            "imgsz": self.imgsz,
            "provider": self.provider,
            "description": self.meta.get("description", ""),
            "metrics": self.meta.get("metrics"),
            "recommended_conf": self.meta.get("recommended_conf"),
            "recommended_iou": self.meta.get("recommended_iou"),
            "count_groups": self.meta.get("count_groups"),
            "max_box_area": self.meta.get("max_box_area"),  # fraction of the image; the page hides larger boxes
            "source": self.meta.get("source", ""),
            "license": self.meta.get("license", ""),
        }

    def preprocess(self, img_bgr: np.ndarray) -> tuple[np.ndarray, LetterboxInfo]:
        lb, info = letterbox(img_bgr, self.imgsz)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), info

    def detect(self, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.45, max_det: int = 300) -> DetectResult:
        t0 = time.perf_counter()
        blob, lb = self.preprocess(img_bgr)
        t1 = time.perf_counter()
        if self.input_fp16:
            blob = blob.astype(np.float16)
        output = self.session.run(None, {self.input_name: blob})[0].astype(np.float32)
        t2 = time.perf_counter()

        boxes, scores, class_ids = decode(output, len(self.names), conf)
        keep = batched_nms(boxes, scores, class_ids, iou)[:max_det]
        h, w = img_bgr.shape[:2]
        result = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            # undo letterbox
            x1 = float(np.clip((x1 - lb.pad_x) / lb.scale, 0, w))
            x2 = float(np.clip((x2 - lb.pad_x) / lb.scale, 0, w))
            y1 = float(np.clip((y1 - lb.pad_y) / lb.scale, 0, h))
            y2 = float(np.clip((y2 - lb.pad_y) / lb.scale, 0, h))
            c = int(class_ids[i])
            label = self.names[c] if c < len(self.names) else str(c)
            result.append(Box(c, label, float(scores[i]), x1, y1, x2, y2))
        t3 = time.perf_counter()

        return DetectResult(
            boxes=result,
            width=w,
            height=h,
            timing_ms={
                "preprocess": round((t1 - t0) * 1000, 2),
                "inference": round((t2 - t1) * 1000, 2),
                "postprocess": round((t3 - t2) * 1000, 2),
                "total": round((t3 - t0) * 1000, 2),
            },
        )
