"""
Application entry point — creates and configures the FastAPI app.

Responsibilities:
  - Lifespan management (model load / unload)
  - Middleware (CORS)
  - Router registration
  - Serves the HTML test page at GET /

Usage:
    uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.app import classifier
from src.app.config import settings
from src.app.routers import health, predict
from src.app.schemas import ErrorDetail

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Path to the HTML template
_TEMPLATE_DIR = Path(__file__).parent / "templates"


# ─── Lifespan ───────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup, release at shutdown."""
    logger.info("Starting up — loading model from %s", settings.model_path)
    classifier.load_model()
    yield
    classifier.clear_model()
    logger.info("Shutdown complete")


# ─── App Factory ────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_title,
    description=(
        "REST API for classifying Raising The Village field check-in images "
        "into 9 operational categories using a fine-tuned YOLO11 model. "
        "Visit the root URL (/) for an interactive test page."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    responses={
        422: {"model": ErrorDetail, "description": "Validation error"},
        500: {"model": ErrorDetail, "description": "Internal server error"},
        503: {"model": ErrorDetail, "description": "Model not loaded"},
    },
)

# ── Middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──
app.include_router(health.router)
app.include_router(predict.router)


# ── HTML Test Page ──
@app.get("/", summary="Interactive test page", include_in_schema=False)
async def index():
    """Serve the HTML frontend for testing image classification."""
    html_path = _TEMPLATE_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Template not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api", summary="API information")
async def api_info():
    """Return service metadata and a list of available endpoints."""
    return {
        "service": settings.app_title,
        "version": settings.app_version,
        "endpoints": {
            "GET  /": "Interactive test page (HTML)",
            "GET  /api": "This endpoint — service info",
            "GET  /health": "Liveness probe",
            "GET  /ready": "Readiness probe",
            "GET  /categories": "List classification categories",
            "POST /predict": "Classify a single image",
            "POST /predict/batch": "Classify multiple images (up to 16)",
            "GET  /docs": "Swagger UI — interactive API documentation",
            "GET  /redoc": "ReDoc — alternative API documentation",
        },
    }


# ─── CLI convenience ───────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
