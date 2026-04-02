# ── Multi-stage Dockerfile for RTV Image Classifier API ──
# Stage 1: install dependencies in a clean layer
# Stage 2: copy source and model, run the API

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level dependencies required by OpenCV and Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Dependency layer (cached unless requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application layer ──
COPY src/ src/

# Copy the trained model weights (mounted or built-in)
# Users should place their best.pt in outputs/training/ before building,
# or mount it at runtime via -v
COPY outputs/training/best.pt outputs/training/best.pt

# Configuration via environment variables
ENV MODEL_PATH=outputs/training/best.pt \
    IMAGE_SIZE=640 \
    MAX_BATCH_SIZE=16

EXPOSE 8000

# Health check for container orchestration (Docker / K8s)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn — 1 worker by default; scale via replicas, not workers
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
