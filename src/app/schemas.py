"""
Pydantic models for API request / response validation.

Every public endpoint uses a typed schema so that:
  - FastAPI auto-generates accurate OpenAPI / Swagger docs
  - Clients get clear error messages on malformed requests
  - Response shape is guaranteed and self-documenting
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionScore(BaseModel):
    """A single class–probability pair in the top-N list."""
    label: str = Field(..., example="compost")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.87)


class PredictionResponse(BaseModel):
    """Result of classifying a single image."""
    predicted_class: str = Field(..., example="compost")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.87)
    description: str = Field(
        ..., example="Composting site for organic waste processing"
    )
    top5: list[PredictionScore] = Field(..., min_length=1, max_length=9)
    inference_time_ms: float = Field(..., example=42.3)


class BatchItem(BaseModel):
    """One result inside a batch response."""
    filename: str
    predicted_class: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    description: str
    top5: list[PredictionScore]


class BatchResponse(BaseModel):
    """Aggregated result for a multi-image batch request."""
    results: list[BatchItem]
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


class ErrorDetail(BaseModel):
    detail: str
