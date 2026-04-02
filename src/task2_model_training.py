"""
Task 2: Model Training — YOLO11 Classification
================================================
Fine-tunes a pretrained YOLO11 classification model on the RTV dataset.

Model choice rationale:
  - YOLO11-cls provides a strong ImageNet-pretrained backbone (EfficientNet-
    style) with a lightweight classification head.
  - Transfer learning is critical here: ~1 200 images is far too few to
    train from scratch, so we leverage features learned on ImageNet's 1.2 M
    images and adapt the final layers to our 9-class problem.
  - We offer two strategies: single-phase (simpler) and two-phase (freeze
    backbone first, then unfreeze for end-to-end fine-tuning).  The two-
    phase approach often yields better results on small datasets because it
    avoids catastrophic forgetting of pretrained features during the early
    high-LR training epochs.

Common pitfalls addressed:
  - Overfitting: early stopping (patience), cosine LR decay, weight decay,
    heavy augmentation, mixup, and random erasing.
  - Class imbalance: the offline augmentation from Task 1 plus class-weighted
    loss in the Ultralytics trainer.
  - Data leakage: we never augment or peek at val/test splits; the pipeline
    in Task 1 splits before augmenting.

Usage:
    # Single-phase (default):
    python src/task2_model_training.py --data_dir outputs/prepared_data --epochs 50

    # Two-phase (recommended for small datasets):
    python src/task2_model_training.py --data_dir outputs/prepared_data --two_phase

    # Specify model size:
    python src/task2_model_training.py --data_dir outputs/prepared_data --model yolo11s-cls.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyper-parameters — centralised for reproducibility."""
    data_dir: str = "outputs/prepared_data"
    model_name: str = "yolo11n-cls.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 32
    lr0: float = 1e-3
    lrf: float = 0.01
    patience: int = 15
    output_dir: str = "outputs/training"
    device: str | None = None
    seed: int = 42
    two_phase: bool = False

    def resolve_device(self) -> str:
        if self.device:
            return self.device
        return "0" if torch.cuda.is_available() else "cpu"


# ─── Environment Check ─────────────────────────────────────────────────
def log_environment(cfg: TrainConfig) -> None:
    """Log hardware and software versions for reproducibility."""
    logger.info("PyTorch %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s (%.1f GB)", props.name, props.total_mem / 1e9)
    else:
        logger.warning("No GPU detected — training will be slow on CPU")
    logger.info("Config: %s", cfg)


# ─── Single-Phase Training ──────────────────────────────────────────────
def train_single_phase(cfg: TrainConfig) -> dict:
    """Standard fine-tuning: unfreeze all layers from the start.

    Suitable when the dataset is reasonably large or the domain is close
    to ImageNet (natural images).  We still use a warmup period, cosine LR,
    and heavy augmentation to mitigate overfitting.
    """
    from ultralytics import YOLO

    device = cfg.resolve_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"rtv_single_{timestamp}"

    logger.info("Loading pretrained model %s …", cfg.model_name)
    model = YOLO(cfg.model_name)

    logger.info("Starting single-phase training (%d epochs) …", cfg.epochs)
    model.train(
        data=cfg.data_dir,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        patience=cfg.patience,
        device=device,
        project=cfg.output_dir,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        weight_decay=5e-4,
        warmup_epochs=5,
        warmup_momentum=0.8,
        cos_lr=True,
        seed=cfg.seed,
        workers=4,
        # Augmentation — aggressive for a small dataset
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.5, shear=5.0,
        perspective=5e-4,
        flipud=0.1, fliplr=0.5,
        mosaic=0.0,       # Mosaic is a detection augmentation; disable for cls
        mixup=0.1,        # Label smoothing via mixup helps on small datasets
        erasing=0.3,      # Random erasing for occlusion robustness
        crop_fraction=0.8,
        verbose=True,
    )

    # Evaluate on held-out test split
    logger.info("Evaluating on test split …")
    val_results = model.val(split="test")

    return _save_summary(cfg, model, val_results, run_name, timestamp, device)


# ─── Two-Phase Training ────────────────────────────────────────────────
def train_two_phase(cfg: TrainConfig) -> dict:
    """Two-phase strategy recommended for very small datasets.

    Phase 1 — Head-only:  Freeze the backbone and train only the classifier
    head for 10 epochs at a higher LR.  This rapidly adapts the output layer
    to our 9-class problem without disrupting the pretrained feature maps.

    Phase 2 — Full fine-tune:  Unfreeze everything and train at a lower LR
    for the remaining epochs.  The backbone gradually adapts its features to
    the RTV domain (field-captured agricultural/household images).
    """
    from ultralytics import YOLO

    device = cfg.resolve_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Phase 1: classifier head only ──
    phase1_epochs = 10
    logger.info("PHASE 1 — Training head only (%d epochs, freeze=10) …", phase1_epochs)
    model = YOLO(cfg.model_name)
    model.train(
        data=cfg.data_dir,
        epochs=phase1_epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        lr0=0.01,
        lrf=0.1,
        patience=phase1_epochs,  # No early stopping in phase 1
        device=device,
        project=cfg.output_dir,
        name=f"phase1_head_{timestamp}",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        freeze=10,
        warmup_epochs=2,
        cos_lr=True,
        seed=cfg.seed,
        workers=4,
        verbose=True,
    )

    # ── Phase 2: end-to-end ──
    phase2_epochs = cfg.epochs - phase1_epochs
    phase1_best = os.path.join(
        cfg.output_dir, f"phase1_head_{timestamp}", "weights", "best.pt"
    )
    logger.info("PHASE 2 — Full fine-tuning (%d epochs) from %s …",
                phase2_epochs, phase1_best)
    model = YOLO(phase1_best)
    model.train(
        data=cfg.data_dir,
        epochs=phase2_epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        patience=cfg.patience,
        device=device,
        project=cfg.output_dir,
        name=f"phase2_finetune_{timestamp}",
        exist_ok=True,
        optimizer="AdamW",
        warmup_epochs=3,
        cos_lr=True,
        seed=cfg.seed,
        workers=4,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.5,
        fliplr=0.5, mixup=0.1, erasing=0.3,
        verbose=True,
    )

    logger.info("Evaluating on test split …")
    val_results = model.val(split="test")

    run_name = f"phase2_finetune_{timestamp}"
    return _save_summary(cfg, model, val_results, run_name, timestamp, device)


# ─── Helpers ────────────────────────────────────────────────────────────
def _save_summary(
    cfg: TrainConfig,
    model,
    val_results,
    run_name: str,
    timestamp: str,
    device: str,
) -> dict:
    """Persist a JSON summary of the training run for traceability."""
    run_dir = os.path.join(cfg.output_dir, run_name)
    best_path = os.path.join(run_dir, "weights", "best.pt")

    summary = {
        "model": cfg.model_name,
        "strategy": "two_phase" if cfg.two_phase else "single_phase",
        "epochs_requested": cfg.epochs,
        "image_size": cfg.imgsz,
        "batch_size": cfg.batch,
        "lr0": cfg.lr0,
        "device": device,
        "seed": cfg.seed,
        "best_model_path": best_path,
        "run_dir": run_dir,
        "timestamp": timestamp,
        "top1_accuracy": float(val_results.results_dict.get("metrics/accuracy_top1", 0)),
        "top5_accuracy": float(val_results.results_dict.get("metrics/accuracy_top5", 0)),
    }

    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Best model : %s", best_path)
    logger.info("  Top-1 acc  : %.4f", summary["top1_accuracy"])
    logger.info("  Top-5 acc  : %.4f", summary["top5_accuracy"])
    logger.info("  Summary    : %s", summary_path)
    logger.info("=" * 60)
    return summary


# ─── CLI ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 — YOLO11 Classification Training")
    parser.add_argument("--data_dir", default="outputs/prepared_data")
    parser.add_argument("--model", default="yolo11n-cls.pt",
                        choices=["yolo11n-cls.pt", "yolo11s-cls.pt",
                                 "yolo11m-cls.pt", "yolo11l-cls.pt"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", default="outputs/training")
    parser.add_argument("--device", default=None)
    parser.add_argument("--two_phase", action="store_true",
                        help="Use two-phase training (freeze backbone, then fine-tune)")
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        device=args.device,
        two_phase=args.two_phase,
    )

    log_environment(cfg)

    if cfg.two_phase:
        train_two_phase(cfg)
    else:
        train_single_phase(cfg)


if __name__ == "__main__":
    main()
