"""Tests run without any trained model: a tiny ONNX graph emits a fixed YOLOv8-style tensor."""
import json

import cv2
import numpy as np
import pytest

from app.cells_classic import count_cells
from app.detector import YoloOnnxDetector, batched_nms, decode, letterbox, nms
from app.imaging import apply_window

IMGSZ = 64
# (4 + nc, N) with nc = 1: columns are candidates [cx, cy, w, h, score]
FAKE_OUT = np.array(
    [
        [32, 33, 10],  # cx
        [32, 32, 10],  # cy
        [20, 20, 8],  # w
        [20, 20, 8],  # h
        [0.9, 0.8, 0.1],  # score
    ],
    dtype=np.float32,
)[None]


def make_fake_model(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    const = numpy_helper.from_array(FAKE_OUT, name="const")
    zero = numpy_helper.from_array(np.zeros((1,), np.float32), name="zero")
    nodes = [
        helper.make_node("ReduceMean", ["images"], ["m"], axes=[1, 2, 3], keepdims=1),  # (1,1,1,1)
        helper.make_node("Reshape", ["m", "shape"], ["m3"]),  # (1,1,1)
        helper.make_node("Mul", ["m3", "zero"], ["z"]),
        helper.make_node("Add", ["z", "const"], ["output0"]),
    ]
    shape = numpy_helper.from_array(np.array([1, 1, 1], np.int64), name="shape")
    graph = helper.make_graph(
        nodes,
        "fake_yolo",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, IMGSZ, IMGSZ])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(FAKE_OUT.shape))],
        initializer=[const, zero, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    path = tmp_path / "fake.onnx"
    onnx.save(model, path)
    (tmp_path / "fake.json").write_text(json.dumps({"names": ["polyp"], "imgsz": IMGSZ}))
    return path


def test_letterbox_geometry():
    img = np.zeros((64, 128, 3), np.uint8)
    out, info = letterbox(img, 64)
    assert out.shape == (64, 64, 3)
    assert info.scale == 0.5 and info.pad_x == 0 and info.pad_y == 16


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_batched_nms_keeps_other_classes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], np.float32)
    scores = np.array([0.9, 0.8], np.float32)
    assert sorted(batched_nms(boxes, scores, np.array([0, 1]), 0.5).tolist()) == [0, 1]


def test_decode_v8_and_v5_layouts():
    boxes, scores, cls = decode(FAKE_OUT, 1, 0.25)
    assert len(boxes) == 2 and boxes[0].tolist() == [22, 22, 42, 42]
    # YOLOv5 layout (1, N, 5 + nc): objectness * class prob
    v5 = np.array([[[32, 32, 20, 20, 0.5, 0.9], [10, 10, 4, 4, 0.1, 0.9]]], np.float32)
    boxes, scores, cls = decode(v5, 1, 0.25)
    assert len(boxes) == 1 and scores[0] == pytest.approx(0.45)


def test_detector_end_to_end(tmp_path):
    det = YoloOnnxDetector(make_fake_model(tmp_path), providers=["CPUExecutionProvider"])
    assert det.names == ["polyp"] and det.imgsz == IMGSZ
    res = det.detect(np.zeros((64, 128, 3), np.uint8), conf=0.25, iou=0.45)
    assert len(res.boxes) == 1  # 2nd box suppressed by NMS, 3rd below conf
    b = res.boxes[0]
    # letterbox space (22..42) -> original image: x / 0.5, (y - 16) / 0.5
    assert (b.x1, b.y1, b.x2, b.y2) == pytest.approx((44, 12, 84, 52))
    assert b.label == "polyp" and b.conf == pytest.approx(0.9)


def test_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    make_fake_model(tmp_path)
    import app.main as main

    monkeypatch.setattr(main, "registry", main.ModelRegistry(tmp_path))
    with TestClient(main.app) as client:
        assert client.get("/api/health").json()["models"] == 1
        assert client.get("/api/models").json()["models"][0]["classes"] == ["polyp"]

        ok, buf = cv2.imencode(".jpg", np.zeros((64, 128, 3), np.uint8))
        r = client.post("/api/detect", files={"file": ("a.jpg", buf.tobytes(), "image/jpeg")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["counts"] == {"polyp": 1} and body["image"] == {"width": 128, "height": 64}

        r = client.post("/api/detect", files={"file": ("a.txt", b"not an image", "text/plain")})
        assert r.status_code == 400
        assert client.post("/api/detect?model=nope", files={"file": ("a.jpg", buf.tobytes())}).status_code == 404

        r = client.post("/api/detect/render", files={"file": ("a.jpg", buf.tobytes(), "image/jpeg")})
        assert r.headers["content-type"] == "image/png"

        with client.websocket_connect("/ws/detect?conf=0.25") as ws:
            ws.send_bytes(buf.tobytes())
            assert len(ws.receive_json()["boxes"]) == 1


def test_window_level():
    px = np.array([-1000, 40, 1000], np.float32)
    assert apply_window(px, center=40, width=400).tolist() == [0, 127, 255]


def test_classic_cell_counter_on_synthetic_smear():
    img = np.full((480, 640, 3), (200, 200, 230), np.uint8)  # pale background
    rbc_centers = [(100, 100), (200, 100), (300, 120), (420, 300), (520, 380)]
    for c in rbc_centers:
        cv2.circle(img, c, 28, (120, 110, 200), -1)  # reddish cells (BGR)
    cv2.circle(img, (300, 320), 40, (160, 40, 110), -1)  # purple WBC
    cv2.circle(img, (560, 80), 6, (160, 40, 110), -1)  # purple platelet
    res = count_cells(img)
    assert res["counts"]["WBC"] == 1
    assert res["counts"]["Platelets"] == 1
    assert res["counts"]["RBC"] == len(rbc_centers)
