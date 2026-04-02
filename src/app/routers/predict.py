"""
Prediction endpoints — single image and batch.

Handles file upload validation, delegates inference to the classifier
module, and maps results to Pydantic response schemas.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.app.classifier import classify_upload, is_model_ready
from src.app.config import settings
from src.app.schemas import (
    BatchItem,
    BatchResponse,
    ErrorDetail,
    PredictionResponse,
    PredictionScore,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


def _require_model() -> None:
    """Raise 503 if the model is not loaded."""
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")


def _validate_extension(filename: str) -> None:
    """Raise 422 if the file extension is not in the allowed set."""
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        allowed = ", ".join(sorted(settings.allowed_extensions))
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a single image",
    responses={
        422: {"model": ErrorDetail, "description": "Invalid file type"},
        503: {"model": ErrorDetail, "description": "Model not loaded"},
    },
)
async def predict(
    file: UploadFile = File(..., description="Image file (jpg, png, webp)"),
):
    """Upload a check-in image and receive the predicted category.

    Returns the top-1 prediction with confidence, a human-readable
    description, and the full top-5 ranking.
    """
    _require_model()
    _validate_extension(file.filename or "unknown.jpg")

    result, elapsed_ms = await classify_upload(file)

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        description=result["description"],
        top5=[PredictionScore(**s) for s in result["top5"]],
        inference_time_ms=elapsed_ms,
    )


@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    summary="Classify multiple images",
    responses={
        422: {"model": ErrorDetail, "description": "Invalid input"},
        503: {"model": ErrorDetail, "description": "Model not loaded"},
    },
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="Up to 16 image files"),
):
    """Upload multiple images in one request.

    Maximum batch size is controlled by the MAX_BATCH_SIZE env variable
    (default 16). Each image is classified independently.
    """
    _require_model()

    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(files)} exceeds limit of {settings.max_batch_size}",
        )

    results: list[BatchItem] = []
    failed = 0
    t0 = time.perf_counter()

    for file in files:
        try:
            _validate_extension(file.filename or "unknown.jpg")
            result, _ = await classify_upload(file)
            results.append(
                BatchItem(
                    filename=file.filename or "unknown",
                    predicted_class=result["predicted_class"],
                    confidence=result["confidence"],
                    description=result["description"],
                    top5=[PredictionScore(**s) for s in result["top5"]],
                )
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Batch item failed (%s): %s", file.filename, exc)
            failed += 1

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return BatchResponse(
        results=results,
        total=len(files),
        failed=failed,
        inference_time_ms=elapsed_ms,
    )
