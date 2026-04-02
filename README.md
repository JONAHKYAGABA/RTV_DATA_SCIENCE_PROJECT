# RTV Check-in Image Classifier

An end-to-end machine learning solution that classifies Raising The Village (RTV) field check-in images into 9 operational categories. Built with YOLO11, FastAPI, and Docker.

**Categories**: compost, goat-sheep-pen, guinea-pig-shelter, liquid-organic, organic, pigsty, poultry-house, tippytap, vsla

---

## Project Structure

```
├── data/                           # Raw dataset (download separately)
├── src/
│   ├── task1_data_analysis.py      # Task 1 — EDA: statistics, plots, quality audit
│   ├── task1_data_pipeline.py      # Task 1 — Data splitting, augmentation, YOLO structure
│   ├── task2_model_training.py     # Task 2 — YOLO11 fine-tuning (single + two-phase)
│   ├── task2_model_evaluation.py   # Task 2 — Metrics, confusion matrix, error analysis
│   └── app/                        # Task 3 — FastAPI application
│       ├── main.py                 #   App factory, lifespan, routes
│       ├── config.py               #   Settings and category metadata
│       ├── schemas.py              #   Pydantic request/response models
│       ├── classifier.py           #   Model loading and inference logic
│       ├── routers/
│       │   ├── health.py           #   /health, /ready, /categories
│       │   └── predict.py          #   /predict, /predict/batch
│       └── templates/
│           └── index.html          #   Interactive test page (drag & drop)
├── docs/
│   └── task1_analysis_writeup.md   # Analysis observations and design decisions
├── outputs/                        # Generated at runtime
│   ├── analysis/                   #   EDA plots and reports
│   ├── prepared_data/              #   Train/val/test splits
│   ├── training/                   #   Model weights and logs
│   └── evaluation/                 #   Evaluation metrics and plots
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA for faster training

### 0. Download the Dataset

The dataset is **not included in this repository**. Download the labelled image dataset and place it in a `data/` folder at the project root, with one subfolder per category:

```
data/
├── compost/            (~150 images)
├── goat-sheep-pen/     (~150 images)
├── guinea-pig-shelter/ (~16 images)
├── liquid-organic/     (~150 images)
├── organic/            (~150 images)
├── pigsty/             (~150 images)
├── poultry-house/      (~150 images)
├── tippytap/           (~150 images)
└── vsla/               (~150 images)
```

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Task 1: Data Analysis & Preparation

### Step 1 — Exploratory Data Analysis

Scan the dataset, generate distribution plots, dimension scatter plots, and a statistical report.

```bash
python src/task1_data_analysis.py --data_dir data/ --output_dir outputs/analysis
```

**Outputs** (in `outputs/analysis/`):
- `class_distribution.png` — Bar chart with imbalance ratio
- `image_dimensions.png` — Width vs height scatter by category
- `file_size_distribution.png` — Box plot per category
- `aspect_ratio_histogram.png` — Informs padding vs crop decision
- `sample_grid.png` — Visual sample of each class
- `analysis_report.txt` — Full statistical summary
- `dataset_metadata.csv` — Per-image metadata table

### Step 2 — Build the Data Pipeline

Perform stratified splitting (70/15/15), compute class weights, and apply offline augmentation to balance the minority class.

```bash
python src/task1_data_pipeline.py --data_dir data/ --output_dir outputs/prepared_data
```

**Options**:
```bash
# Skip augmentation (for debugging)
python src/task1_data_pipeline.py --data_dir data/ --no_augment

# Custom seed
python src/task1_data_pipeline.py --data_dir data/ --seed 123
```

**Outputs** (in `outputs/prepared_data/`):
- `train/`, `val/`, `test/` — YOLO-compatible directory structure
- `dataset.yaml` — YOLO training config
- `class_weights.json` — Inverse-frequency weights for loss function

### Analysis Write-up

See [docs/task1_analysis_writeup.md](docs/task1_analysis_writeup.md) for detailed observations, design decisions, and trade-off analysis.

---

## Task 2: Model Training & Evaluation

### Step 3 — Train the Model

Fine-tune a pretrained YOLO11 classification model on the prepared data.

```bash
# Single-phase training (default — all layers unfrozen)
python src/task2_model_training.py \
    --data_dir outputs/prepared_data \
    --epochs 50

# Two-phase training (recommended for small datasets)
# Phase 1: freeze backbone, train head (10 epochs)
# Phase 2: unfreeze all, fine-tune (remaining epochs)
python src/task2_model_training.py \
    --data_dir outputs/prepared_data \
    --epochs 50 \
    --two_phase
```

**Options**:
```bash
--model yolo11s-cls.pt   # Larger model (n/s/m/l variants available)
--batch 16               # Reduce if GPU memory is limited
--imgsz 640              # Input image size
--lr 0.001               # Initial learning rate
--patience 15            # Early stopping patience
--device cpu             # Force CPU (auto-detects GPU by default)
```

**Outputs** (in `outputs/training/<run_name>/`):
- `weights/best.pt` — Best model checkpoint
- `weights/last.pt` — Final epoch checkpoint
- `training_summary.json` — Hyperparameters and final metrics
- Training curves and logs generated by Ultralytics

### Step 4 — Evaluate the Model

Run comprehensive evaluation on the held-out test set.

```bash
python src/task2_model_evaluation.py \
    --model outputs/training/<run_name>/weights/best.pt \
    --data_dir outputs/prepared_data
```

**Outputs** (in `outputs/evaluation/`):
- `confusion_matrix.png` — Raw counts + normalised heatmap
- `per_class_f1.png` — F1 scores per category
- `confidence_histogram.png` — Correct vs incorrect confidence distributions
- `misclassification_audit.txt` — Error taxonomy with high-confidence errors
- `metrics.json` — Full metrics for programmatic comparison
- `predictions.json` — Per-image predictions for drill-down analysis

---

## Task 3: API & Deployment

The API is structured as a proper Python package (`src/app/`) with separated concerns:

| Module | Responsibility |
|--------|---------------|
| `main.py` | App factory, lifespan, middleware, route registration |
| `config.py` | Settings from environment variables, category metadata |
| `schemas.py` | Pydantic models for all request/response types |
| `classifier.py` | Model loading and inference (decoupled from HTTP) |
| `routers/health.py` | Health, readiness, and category endpoints |
| `routers/predict.py` | Single and batch prediction endpoints |
| `templates/index.html` | Interactive web UI for testing |

### Step 5 — Run the API Locally

```bash
# Set the model path (Linux/Mac)
export MODEL_PATH=outputs/training/<run_name>/weights/best.pt

# Set the model path (Windows PowerShell)
$env:MODEL_PATH = "outputs/training/<run_name>/weights/best.pt"

# Start the server
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser:
- **http://localhost:8000** — Interactive test page (drag & drop images)
- **http://localhost:8000/docs** — Swagger API documentation
- **http://localhost:8000/redoc** — ReDoc API documentation

### Step 6 — Test with the HTML Page

You can test the API in **two ways**:

**Option A — Served by FastAPI (recommended):**
Open `http://localhost:8000` in your browser. The test page is served directly by the API, so everything works automatically.

**Option B — Open the HTML file directly from disk:**
Double-click `src/app/templates/index.html` to open it in your browser.
The page will detect it's running from a `file://` URL and default to `http://localhost:8000` as the API address. You can change the API URL in the input field at the top of the page if the server is on a different host/port.

> **Note:** CORS is configured to accept requests from `null` origins (which is what browsers send for `file://` pages), so this works out of the box.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive HTML test page |
| `GET` | `/api` | Service metadata and endpoint list |
| `GET` | `/health` | Liveness probe (always 200 if running) |
| `GET` | `/ready` | Readiness probe (200 only when model loaded) |
| `GET` | `/categories` | List all 9 categories with descriptions |
| `POST` | `/predict` | Classify a single uploaded image |
| `POST` | `/predict/batch` | Classify up to 16 images in one request |
| `GET` | `/docs` | Interactive Swagger documentation |
| `GET` | `/redoc` | ReDoc documentation |

### Example: Classify an Image

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/image.jpg"
```

**Response**:
```json
{
  "predicted_class": "compost",
  "confidence": 0.873,
  "description": "Composting site for organic waste processing",
  "top5": [
    {"label": "compost", "confidence": 0.873},
    {"label": "organic", "confidence": 0.064},
    {"label": "liquid-organic", "confidence": 0.031},
    {"label": "tippytap", "confidence": 0.012},
    {"label": "pigsty", "confidence": 0.009}
  ],
  "inference_time_ms": 42.3
}
```

### Example: Batch Classification

```bash
curl -X POST http://localhost:8000/predict/batch \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg" \
     -F "files=@image3.jpg"
```

---

## Docker Deployment

### Build and Run

```bash
# Copy your trained model to the expected location
mkdir -p outputs/training
cp outputs/training/<run_name>/weights/best.pt outputs/training/best.pt

# Build the Docker image
docker build -t rtv-classifier .

# Run the container
docker run -p 8000:8000 rtv-classifier
```

### Using Docker Compose (recommended)

```bash
# Mount model weights from host — no need to bake into image
docker compose up --build
```

Then visit:
- **http://localhost:8000** — Interactive test page
- **http://localhost:8000/docs** — Swagger docs

### Configuration via Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `outputs/training/best.pt` | Path to YOLO model weights |
| `IMAGE_SIZE` | `640` | Input image size for inference |
| `MAX_BATCH_SIZE` | `16` | Maximum files in batch endpoint |

### Health Checks

The Docker container includes a built-in health check. For Kubernetes:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  periodSeconds: 30
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  periodSeconds: 10
```

---

## Complete Pipeline (End-to-End)

Run all tasks sequentially:

```bash
# 1. Install
pip install -r requirements.txt

# 2. Analyse
python src/task1_data_analysis.py --data_dir data/

# 3. Prepare data
python src/task1_data_pipeline.py --data_dir data/

# 4. Train (use --two_phase for best results on small data)
python src/task2_model_training.py --data_dir outputs/prepared_data --two_phase

# 5. Evaluate (replace <run_name> with actual folder)
python src/task2_model_evaluation.py \
    --model outputs/training/<run_name>/weights/best.pt

# 6. Deploy
export MODEL_PATH=outputs/training/<run_name>/weights/best.pt
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

---

## Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| **YOLO11-cls** over ResNet/EfficientNet | Built-in augmentation pipeline, fast inference, single-framework training + deployment |
| **Transfer learning** (not training from scratch) | ~1,200 images is far too few; pretrained ImageNet features provide strong initialisation |
| **Two-phase training** | Protects pretrained features from early high-LR updates; demonstrated improvement on small datasets |
| **Offline + online augmentation** | Offline for class balancing (visible, auditable); online for epoch-level variety |
| **Macro F1** as primary metric | Weights every class equally — exposes minority-class weakness that accuracy hides |
| **FastAPI** (not Flask/Gradio) | Async support, automatic OpenAPI docs, Pydantic validation, production-grade |
| **Modular app structure** | Separated config / schemas / classifier / routers — testable, maintainable, standard practice |
| **HTML test page at /** | Allows non-technical users to test the classifier without curl or Swagger |
| **Docker** with health checks | Production-ready containerisation with orchestration support |
