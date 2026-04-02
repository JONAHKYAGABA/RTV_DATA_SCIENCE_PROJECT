"""
Model loading and inference logic.

Separated from the HTTP layer so it can be:
  - unit-tested independently
  - swapped for a different backend (ONNX, TensorRT) without touching routes
  - called from CLI scripts as well as the API
"""

from __future__ import annotations

import logging
import tempfile
import os
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import UploadFile

from src.app.config import CATEGORIES, settings

logger = logging.getLogger(__name__)

# Module-level model reference — set by load_model / clear_model
_model = None


def load_model() -> None:
    """Load the YOLO model from disk into memory."""
    global _model
    model_path = settings.model_path

    if not Path(model_path).exists():
        logger.warning(
            "Model file not found at %s — API will start in degraded mode",
            model_path,
        )
        return

    from ultralytics import YOLO
    _model = YOLO(model_path)
    logger.info("Model loaded from %s", model_path)


def clear_model() -> None:
    """Release the model from memory."""
    global _model
    _model = None
    logger.info("Model unloaded")


def is_model_ready() -> bool:
    """Check whether the model is loaded and ready for inference."""
    return _model is not None


def classify_image(image_path: str) -> dict[str, Any]:
    """Run inference on a single image file.

    Returns a dict with keys: predicted_class, confidence, description, top5.
    Raises RuntimeError if the model is not loaded.
    """
    if _model is None:
        raise RuntimeError("Model not loaded")

    preds = _model.predict(image_path, imgsz=settings.image_size, verbose=False)
    pred = preds[0]

    top1_idx = pred.probs.top1
    top1_conf = float(pred.probs.top1conf)
    predicted_class = pred.names[top1_idx]

    top5 = []
    for idx in pred.probs.top5:
        label = pred.names[idx]
        conf = float(pred.probs.data[idx])
        top5.append({"label": label, "confidence": round(conf, 4)})

    return {
        "predicted_class": predicted_class,
        "confidence": round(top1_conf, 4),
        "description": CATEGORIES.get(predicted_class, ""),
        "top5": top5,
    }


async def classify_upload(upload: UploadFile) -> tuple[dict[str, Any], float]:
    """Save an UploadFile to a temp path, classify it, and clean up.

    Returns (result_dict, inference_time_ms).
    """
    suffix = Path(upload.filename or "image.jpg").suffix
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        t0 = time.perf_counter()
        result = classify_image(tmp_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return result, round(elapsed_ms, 1)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
