"""
Task 1: Data Loading Pipeline
===============================
Builds a reproducible data pipeline that:
  1. Performs stratified train / validation / test splitting (70/15/15)
  2. Applies offline augmentation to balance the minority class
  3. Computes inverse-frequency class weights for loss weighting
  4. Outputs a YOLO-compatible directory structure with a dataset.yaml

Design decisions:
  - Stratified splitting ensures proportional representation even for the
    16-image guinea-pig-shelter class.
  - Offline augmentation (rather than only online) lets us visually inspect
    the synthetic images and guarantees the minority class reaches a usable
    training volume before training begins.
  - We target 80% of the majority class size — full equalisation on such a
    small minority risks memorising augmentation artefacts.

Usage:
    python src/task1_data_pipeline.py --data_dir data/ --output_dir outputs/prepared_data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────
_IMG_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_RANDOM_SEED = 42


@dataclass
class PipelineConfig:
    """All tuneable pipeline parameters in one place."""
    data_dir: str = "data/"
    output_dir: str = "outputs/prepared_data"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    target_balance: float = 0.80      # Oversample minority to this fraction of majority
    augment: bool = True
    seed: int = _RANDOM_SEED

    def __post_init__(self) -> None:
        assert abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"


# ─── Data Collection ────────────────────────────────────────────────────
def collect_samples(data_dir: str) -> list[dict[str, str]]:
    """Return a list of {"path": ..., "label": ...} for every valid image."""
    samples: list[dict[str, str]] = []
    data_path = Path(data_dir)

    for category_dir in sorted(data_path.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith((".", "_")):
            continue
        label = category_dir.name
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() in _IMG_EXTENSIONS:
                samples.append({"path": str(img_file), "label": label})

    logger.info("Collected %d images across %d categories",
                len(samples), len({s["label"] for s in samples}))
    return samples


# ─── Splitting ──────────────────────────────────────────────────────────
def stratified_split(
    samples: list[dict[str, str]],
    cfg: PipelineConfig,
) -> dict[str, list[tuple[str, str]]]:
    """Stratified three-way split preserving class proportions.

    Returns a dict mapping split name -> list of (path, label) tuples.
    """
    paths = [s["path"] for s in samples]
    labels = [s["label"] for s in samples]

    # Two-step split: (train+val) vs test, then train vs val
    train_val_p, test_p, train_val_l, test_l = train_test_split(
        paths, labels,
        test_size=cfg.test_ratio,
        stratify=labels,
        random_state=cfg.seed,
    )
    val_frac = cfg.val_ratio / (cfg.train_ratio + cfg.val_ratio)
    train_p, val_p, train_l, val_l = train_test_split(
        train_val_p, train_val_l,
        test_size=val_frac,
        stratify=train_val_l,
        random_state=cfg.seed,
    )

    splits = {
        "train": list(zip(train_p, train_l)),
        "val": list(zip(val_p, val_l)),
        "test": list(zip(test_p, test_l)),
    }

    for name, data in splits.items():
        counts = Counter(lbl for _, lbl in data)
        logger.info("  %s: %d images — %s", name, len(data),
                     ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return splits


# ─── Augmentation ───────────────────────────────────────────────────────
# Each function takes a PIL Image and returns a new PIL Image.
# We compose them deterministically so results are reproducible.

def _flip_horizontal(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def _rotate_small(img: Image.Image) -> Image.Image:
    angle = random.uniform(-15, 15)
    return img.rotate(angle, fillcolor=(0, 0, 0))

def _rotate_large(img: Image.Image) -> Image.Image:
    angle = random.uniform(-30, 30)
    return img.rotate(angle, fillcolor=(0, 0, 0))

def _brightness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))

def _contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

def _saturation(img: Image.Image) -> Image.Image:
    return ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))

def _blur(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

def _sharpness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(random.uniform(0.5, 2.0))

def _random_crop(img: Image.Image) -> Image.Image:
    """Crop 80-100 % of the image and resize back to the original dimensions."""
    w, h = img.size
    frac = random.uniform(0.80, 1.0)
    nw, nh = int(w * frac), int(h * frac)
    left = random.randint(0, w - nw)
    top = random.randint(0, h - nh)
    return img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)

def _combined(img: Image.Image) -> Image.Image:
    """Compose several light transforms — mimics real field variation."""
    if random.random() > 0.5:
        img = _flip_horizontal(img)
    img = img.rotate(random.uniform(-10, 10), fillcolor=(0, 0, 0))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img


_AUGMENTATIONS = [
    _flip_horizontal, _rotate_small, _rotate_large,
    _brightness, _contrast, _saturation,
    _blur, _sharpness, _random_crop, _combined,
]


def augment_image(img: Image.Image, index: int) -> Image.Image:
    """Apply the i-th augmentation (cycles through the augmentation pool)."""
    fn = _AUGMENTATIONS[index % len(_AUGMENTATIONS)]
    return fn(img)


# ─── Class-Imbalance Handling ───────────────────────────────────────────
def compute_augmentation_plan(
    train_data: list[tuple[str, str]],
    target_fraction: float,
) -> dict[str, int]:
    """Determine how many synthetic images each class needs.

    Returns a dict of label -> number_of_augmented_images_to_create.
    Classes above the target are left unchanged.
    """
    counts = Counter(lbl for _, lbl in train_data)
    target = int(max(counts.values()) * target_fraction)

    plan: dict[str, int] = {}
    for label, count in counts.items():
        plan[label] = max(0, target - count)
    return plan


def compute_class_weights(train_data: list[tuple[str, str]]) -> dict[str, float]:
    """Inverse-frequency weighting: w_c = N / (C * n_c).

    These weights are saved alongside the dataset so the training script
    can apply them in the loss function. Using inverse-frequency rather
    than equal weights ensures the model is penalised proportionally more
    for errors on the minority class.
    """
    labels = [lbl for _, lbl in train_data]
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)

    weights = {label: total / (n_classes * cnt) for label, cnt in counts.items()}
    logger.info("Class weights: %s",
                ", ".join(f"{k}={v:.3f}" for k, v in sorted(weights.items())))
    return weights


# ─── Pipeline Orchestration ─────────────────────────────────────────────
def build_dataset(splits: dict, cfg: PipelineConfig) -> None:
    """Copy images into train/val/test folders and apply augmentation."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Copy originals
    for split_name, data in splits.items():
        for img_path, label in tqdm(data, desc=f"Copying {split_name}", leave=False):
            dest_dir = os.path.join(cfg.output_dir, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

    # Augment training set only
    if not cfg.augment:
        return

    plan = compute_augmentation_plan(splits["train"], cfg.target_balance)
    for label, n_needed in plan.items():
        if n_needed == 0:
            continue
        logger.info("Augmenting %s: +%d synthetic images", label, n_needed)
        source_images = [p for p, l in splits["train"] if l == label]
        dest_dir = os.path.join(cfg.output_dir, "train", label)

        for i in range(n_needed):
            src = random.choice(source_images)
            try:
                img = Image.open(src).convert("RGB")
                aug = augment_image(img, i)
                aug.save(os.path.join(dest_dir, f"aug_{i:04d}_{os.path.basename(src)}"),
                         quality=95)
            except Exception as exc:
                logger.warning("Augmentation failed for %s: %s", src, exc)


def write_dataset_yaml(cfg: PipelineConfig, classes: list[str]) -> str:
    """Create a YOLO-compatible dataset.yaml."""
    yaml_lines = [
        "# RTV Check-in Image Classification Dataset",
        f"# Generated by task1_data_pipeline.py (seed={cfg.seed})",
        "",
        f"path: {os.path.abspath(cfg.output_dir)}",
        "train: train",
        "val: val",
        "test: test",
        "",
        f"nc: {len(classes)}",
        "",
        "names:",
    ]
    for i, cls in enumerate(classes):
        yaml_lines.append(f"  {i}: {cls}")

    yaml_path = os.path.join(cfg.output_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(yaml_lines) + "\n")
    logger.info("Dataset YAML written to %s", yaml_path)
    return yaml_path


def print_final_counts(output_dir: str) -> None:
    """Log the final image counts per split and class."""
    for split in ("train", "val", "test"):
        split_dir = Path(output_dir) / split
        if not split_dir.exists():
            continue
        total = 0
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                n = len(list(cls_dir.glob("*")))
                total += n
                logger.info("  %s/%s: %d", split, cls_dir.name, n)
        logger.info("  %s total: %d", split, total)


# ─── Entry Point ────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1 — Data Pipeline")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="outputs/prepared_data")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--seed", type=int, default=_RANDOM_SEED)
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        augment=not args.no_augment,
        seed=args.seed,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DATA PIPELINE — RTV Check-in Image Classification")
    logger.info("=" * 60)

    # 1 — Collect
    samples = collect_samples(cfg.data_dir)

    # 2 — Split
    logger.info("Splitting with ratios %.0f/%.0f/%.0f …",
                cfg.train_ratio * 100, cfg.val_ratio * 100, cfg.test_ratio * 100)
    splits = stratified_split(samples, cfg)

    # 3 — Class weights
    weights = compute_class_weights(splits["train"])
    weights_path = os.path.join(cfg.output_dir, "class_weights.json")
    with open(weights_path, "w") as fh:
        json.dump(weights, fh, indent=2)

    # 4 — Build directory + augment
    build_dataset(splits, cfg)

    # 5 — YAML config
    classes = sorted({s["label"] for s in samples})
    write_dataset_yaml(cfg, classes)

    # 6 — Summary
    print_final_counts(cfg.output_dir)
    logger.info("Pipeline complete — output at %s", cfg.output_dir)


if __name__ == "__main__":
    main()
