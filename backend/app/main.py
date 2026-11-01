"""MedLens FastAPI service.

Endpoints
  GET  /api/health
  GET  /api/models
  POST /api/detect          image/DICOM -> boxes JSON
  POST /api/detect/render   image/DICOM -> annotated PNG
  POST /api/cells/classic   traditional OpenCV blood-cell counting
  WS   /ws/detect           binary JPEG frames in, boxes JSON out (video mode)
  /samples/...              demo images / videos (manifest.json lists them per imaging menu)
  /                         static frontend
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .cells_classic import count_cells
from .imaging import ImageDecodeError, decode_image, encode_png
from .registry import ModelRegistry
from .render import draw_boxes

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.getenv("MEDLENS_MODELS_DIR", ROOT / "models"))
FRONTEND_DIR = Path(os.getenv("MEDLENS_FRONTEND_DIR", ROOT / "frontend"))
SAMPLES_DIR = Path(os.getenv("MEDLENS_SAMPLES_DIR", ROOT / "samples"))
MAX_UPLOAD_MB = float(os.getenv("MEDLENS_MAX_UPLOAD_MB", "30"))
BACKEND_NAME = "python-onnxruntime"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
registry = ModelRegistry(MODELS_DIR, num_threads=int(os.getenv("MEDLENS_ORT_THREADS", "0")))


@asynccontextmanager
async def lifespan(_: FastAPI):
    registry.load_all()
    yield


app = FastAPI(title="MedLens", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _read_upload(file: UploadFile) -> bytes:
    data = file.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
    return data


def _get_model(name: str | None):
    det = registry.get(name)
    if det is None:
        available = [m["name"] for m in registry.list()]
        msg = f"model '{name}' not found; available: {available}" if available else (
            "no model loaded - train and export one into models/ (see docs/DESIGN.md)"
        )
        raise HTTPException(404, msg)
    return det


def _decode(data: bytes, filename: str, wc, ww):
    try:
        return decode_image(data, filename, wc, ww)
    except ImageDecodeError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/health")
def health():
    body = {"status": "ok", "backend": BACKEND_NAME, "version": __version__, "models": len(registry.list())}
    if os.getenv("MEDLENS_CPP_PORT"):  # tells the page where the C++ backend listens (same host)
        body["cpp_port"] = int(os.environ["MEDLENS_CPP_PORT"])
    return body


@app.get("/api/models")
def models():
    return {"backend": BACKEND_NAME, "models": registry.list()}


@app.post("/api/detect")
def detect(
    file: UploadFile = File(...),
    model: str | None = Query(None),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    wc: float | None = Query(None, description="DICOM window center"),
    ww: float | None = Query(None, description="DICOM window width"),
):
    det = _get_model(model)
    data = _read_upload(file)
    t0 = time.perf_counter()
    img, dicom = _decode(data, file.filename or "", wc, ww)
    decode_ms = (time.perf_counter() - t0) * 1000
    res = det.detect(img, conf=conf, iou=iou)

    body = {
        "backend": BACKEND_NAME,
        "model": det.name,
        "provider": det.provider,
        "image": {"width": res.width, "height": res.height},
        "boxes": [b.to_dict() for b in res.boxes],
        "counts": _counts(res.boxes),
        "timing_ms": {"decode": round(decode_ms, 2), **res.timing_ms},
    }
    if dicom is not None:  # browsers cannot show DICOM: send back the windowed image
        body["dicom"] = dicom
        body["image_png_base64"] = base64.b64encode(encode_png(img)).decode()
    return body


@app.post("/api/detect/render")
def detect_render(
    file: UploadFile = File(...),
    model: str | None = Query(None),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    iou: float = Query(0.45, ge=0.0, le=1.0),
    wc: float | None = None,
    ww: float | None = None,
):
    det = _get_model(model)
    img, _ = _decode(_read_upload(file), file.filename or "", wc, ww)
    res = det.detect(img, conf=conf, iou=iou)
    png = encode_png(draw_boxes(img, [b.to_dict() for b in res.boxes]))
    return Response(png, media_type="image/png")


@app.post("/api/cells/classic")
def cells_classic(file: UploadFile = File(...)):
    img, _ = _decode(_read_upload(file), file.filename or "", None, None)
    return count_cells(img)


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket):
    """Video mode. Client sends JPEG frames (binary) and optional JSON text {"conf":..,"iou":..,"model":..}."""
    await ws.accept()
    params = {
        "model": ws.query_params.get("model"),
        "conf": float(ws.query_params.get("conf", 0.25)),
        "iou": float(ws.query_params.get("iou", 0.45)),
    }
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text"):
                params.update(json.loads(msg["text"]))
                continue
            data = msg.get("bytes")
            if not data:
                continue
            det = registry.get(params["model"])
            if det is None:
                await ws.send_json({"error": "model not found"})
                continue
            try:
                img, _ = decode_image(data)
            except ImageDecodeError as e:
                await ws.send_json({"error": str(e)})
                continue
            res = await run_in_threadpool(det.detect, img, float(params["conf"]), float(params["iou"]))
            await ws.send_json(
                {
                    "model": det.name,
                    "image": {"width": res.width, "height": res.height},
                    "boxes": [b.to_dict() for b in res.boxes],
                    "counts": _counts(res.boxes),
                    "timing_ms": res.timing_ms,
                }
            )
    except WebSocketDisconnect:
        pass


def _counts(boxes) -> dict:
    counts: dict[str, int] = {}
    for b in boxes:
        counts[b.label] = counts.get(b.label, 0) + 1
    return counts


if SAMPLES_DIR.exists():  # demo images / videos + manifest.json (scripts/fetch_samples.py)
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
