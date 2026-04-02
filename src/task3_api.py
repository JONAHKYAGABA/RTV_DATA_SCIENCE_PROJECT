"""
Task 3: Production REST API — FastAPI
=======================================
A well-structured classification API built with FastAPI that exposes the
trained YOLO model for single-image and batch inference.

Key design decisions:
  - Pydantic models for request/response validation and OpenAPI docs
  - Async endpoints for non-blocking I/O under concurrent load
  - Proper HTTP status codes and structured error responses
  - Health / readiness endpoints for container orchestration
  - Model loaded once at startup (singleton), not per-request
  - Temp file cleanup in finally blocks to avoid disk leaks
  - CORS middleware enabled for frontend integration

Endpoints:
  GET  /                → API info & available endpoints
  GET  /health          → Liveness probe
  GET  /ready           → Readiness probe (model loaded?)
  GET  /categories      → List of class labels with descriptions
  POST /predict         → Classify a single uploaded image
  POST /predict/batch   → Classify multiple images in one request

Usage:
    uvicorn src.task3_api:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Configuration ──────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "outputs/training/best.pt")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "640"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "16"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ─── Category metadata ─────────────────────────────────────────────────
CATEGORIES: dict[str, str] = {
    "compost": "Composting site for organic waste processing",
    "goat-sheep-pen": "Enclosure for goats or sheep",
    "guinea-pig-shelter": "Shelter structure for guinea pigs",
    "liquid-organic": "Liquid organic fertilizer setup",
    "organic": "Organic farming / gardening installation",
    "pigsty": "Pig housing structure",
    "poultry-house": "Poultry / chicken housing",
    "tippytap": "Handwashing station (tippy-tap)",
    "vsla": "Village Savings and Loan Association meeting",
}


# ─── Pydantic Schemas ──────────────────────────────────────────────────
class PredictionScore(BaseModel):
    """A single class-probability pair."""
    label: str = Field(..., example="compost")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.87)


class PredictionResponse(BaseModel):
    """Response for a single image classification."""
    predicted_class: str = Field(..., example="compost")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.87)
    description: str = Field(..., example="Composting site for organic waste processing")
    top5: list[PredictionScore] = Field(..., min_length=1, max_length=9)
    inference_time_ms: float = Field(..., example=42.3)


class BatchPredictionItem(BaseModel):
    """One result inside a batch response."""
    filename: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    top5: list[PredictionScore]


class BatchPredictionResponse(BaseModel):
    """Response for batch image classification."""
    results: list[BatchPredictionItem]
    total: int
    failed: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    model_loaded: bool
    model_path: str


class CategoryItem(BaseModel):
    label: str
    description: str


class ErrorResponse(BaseModel):
    detail: str


# ─── Model Singleton ───────────────────────────────────────────────────
_model = None


def get_model():
    """Return the loaded model; raise 503 if not ready."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release at shutdown."""
    global _model
    logger.info("Loading model from %s …", MODEL_PATH)
    if not Path(MODEL_PATH).exists():
        logger.warning("Model file not found at %s — API will start in degraded mode", MODEL_PATH)
    else:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully")
    yield
    _model = None
    logger.info("Model unloaded")


# ─── FastAPI Application ───────────────────────────────────────────────
app = FastAPI(
    title="RTV Check-in Image Classifier",
    description=(
        "REST API for classifying Raising The Village field check-in images "
        "into 9 operational categories using a fine-tuned YOLO11 model."
    ),
    version="1.0.0",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helper ─────────────────────────────────────────────────────────────
def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )


def _classify(model, image_path: str) -> dict:
    """Run YOLO inference on a single image file and return structured results."""
    preds = model.predict(image_path, imgsz=IMAGE_SIZE, verbose=False)
    pred = preds[0]

    top1_idx = pred.probs.top1
    top1_conf = float(pred.probs.top1conf)
    predicted_class = pred.names[top1_idx]

    top5 = []
    for idx in pred.probs.top5:
        name = pred.names[idx]
        conf = float(pred.probs.data[idx])
        top5.append({"label": name, "confidence": round(conf, 4)})

    return {
        "predicted_class": predicted_class,
        "confidence": round(top1_conf, 4),
        "description": CATEGORIES.get(predicted_class, ""),
        "top5": top5,
    }


# ─── Endpoints ──────────────────────────────────────────────────────────

@app.get("/", summary="API information")
async def root():
    """Return basic API info and a list of available endpoints."""
    return {
        "service": "RTV Check-in Image Classifier",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Liveness probe",
            "GET /ready": "Readiness probe",
            "GET /categories": "List classification categories",
            "POST /predict": "Classify a single image",
            "POST /predict/batch": "Classify multiple images",
            "GET /docs": "Interactive API documentation (Swagger)",
        },
    }


@app.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    """Basic health check — always returns 200 if the process is running."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_path=MODEL_PATH,
    )


@app.get("/ready", summary="Readiness probe")
async def ready():
    """Returns 200 only when the model is loaded and ready for inference."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/categories", response_model=list[CategoryItem], summary="List categories")
async def list_categories():
    """Return all 9 classification categories with descriptions."""
    return [CategoryItem(label=k, description=v) for k, v in CATEGORIES.items()]


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify a single image",
    responses={422: {"model": ErrorResponse}},
)
async def predict(file: UploadFile = File(..., description="Image file (jpg, png, webp)")):
    """Upload a single check-in image and receive a classification result.

    The response includes the top prediction, its confidence score,
    a human-readable description of the category, and the top-5 predictions
    with their confidence scores.
    """
    _validate_extension(file.filename or "unknown.jpg")
    model = get_model()

    # Write to a temp file — YOLO expects a filesystem path
    tmp_path: Optional[str] = None
    try:
        suffix = Path(file.filename or "img.jpg").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        t0 = time.perf_counter()
        result = _classify(model, tmp_path)
        elapsed_ms = (time.perf_counter() - t0) * 1000

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return PredictionResponse(
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        description=result["description"],
        top5=[PredictionScore(**s) for s in result["top5"]],
        inference_time_ms=round(elapsed_ms, 1),
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Classify multiple images",
    responses={422: {"model": ErrorResponse}},
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="Up to 16 image files"),
):
    """Upload multiple images and receive classification results for each.

    Maximum batch size is controlled by the MAX_BATCH_SIZE environment
    variable (default: 16).
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(files)} exceeds maximum of {MAX_BATCH_SIZE}",
        )

    model = get_model()
    results: list[BatchPredictionItem] = []
    failed = 0

    t0 = time.perf_counter()
    for file in files:
        tmp_path: Optional[str] = None
        try:
            _validate_extension(file.filename or "unknown.jpg")
            suffix = Path(file.filename or "img.jpg").suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            result = _classify(model, tmp_path)
            results.append(BatchPredictionItem(
                filename=file.filename or "unknown",
                predicted_class=result["predicted_class"],
                confidence=result["confidence"],
                description=result["description"],
                top5=[PredictionScore(**s) for s in result["top5"]],
            ))
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Batch item failed (%s): %s", file.filename, exc)
            failed += 1
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return BatchPredictionResponse(
        results=results,
        total=len(files),
        failed=failed,
        inference_time_ms=round(elapsed_ms, 1),
    )


# ─── CLI convenience ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.task3_api:app", host="0.0.0.0", port=8000, reload=True)
