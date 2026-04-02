"""
Application configuration.

All settings are read from environment variables with sensible defaults,
making the app configurable without code changes (12-factor style).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Immutable application settings — populated once at import time."""

    model_path: str = os.getenv("MODEL_PATH", "outputs/training/best.pt")
    image_size: int = int(os.getenv("IMAGE_SIZE", "640"))
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "16"))
    allowed_extensions: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    app_title: str = "RTV Check-in Image Classifier"
    app_version: str = "1.0.0"


# Singleton — import this wherever settings are needed
settings = Settings()


# Category labels and human-readable descriptions
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
