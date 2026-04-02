"""
Health, readiness, and category listing endpoints.

These are lightweight read-only endpoints used by:
  - Container orchestrators (K8s, Docker) for liveness / readiness probes
  - Frontend clients to discover available categories
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.app.classifier import is_model_ready
from src.app.config import CATEGORIES, settings
from src.app.schemas import CategoryItem, HealthResponse

router = APIRouter(tags=["Health & Info"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health():
    """Always returns 200 if the process is alive."""
    return HealthResponse(
        status="healthy",
        model_loaded=is_model_ready(),
        model_path=settings.model_path,
    )


@router.get("/ready", summary="Readiness probe")
async def ready():
    """Returns 200 only when the model is loaded and ready for inference."""
    if not is_model_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@router.get(
    "/categories",
    response_model=list[CategoryItem],
    summary="List classification categories",
)
async def list_categories():
    """Return all 9 RTV check-in categories with descriptions."""
    return [CategoryItem(label=k, description=v) for k, v in CATEGORIES.items()]
