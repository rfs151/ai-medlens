"""Loads every models/*.onnx at startup and keeps one detector per model."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from .detector import YoloOnnxDetector

log = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, models_dir: str | Path, num_threads: int = 0):
        self.models_dir = Path(models_dir)
        self.num_threads = num_threads
        self._models: dict[str, YoloOnnxDetector] = {}
        self._lock = threading.Lock()

    def load_all(self) -> None:
        if not self.models_dir.exists():
            log.warning("models dir %s not found", self.models_dir)
            return
        for path in sorted(self.models_dir.glob("*.onnx")):
            try:
                det = YoloOnnxDetector(path, num_threads=self.num_threads)
                with self._lock:
                    self._models[det.name] = det
                log.info("loaded model %s (%s, %d classes, %s)", det.name, det.provider, len(det.names), det.imgsz)
            except Exception:  # keep serving other models
                log.exception("failed to load %s", path)

    def get(self, name: str | None) -> YoloOnnxDetector | None:
        with self._lock:
            if name:
                return self._models.get(name)
            return next(iter(self._models.values()), None)

    def list(self) -> list[dict]:
        with self._lock:
            return [m.info() for m in self._models.values()]
