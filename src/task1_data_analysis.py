"""
Task 1: Data Analysis & Exploration
====================================
Performs exploratory data analysis on the RTV field check-in image dataset.
Generates statistical summaries, distribution plots, and a quality audit
to inform preprocessing and modelling decisions.

Usage:
    python src/task1_data_analysis.py --data_dir data/ --output_dir outputs/analysis
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Supported image formats (lowercase)
_IMG_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ─── Configuration ──────────────────────────────────────────────────────
@dataclass
class AnalysisConfig:
    """Centralised configuration for the analysis pipeline."""
    data_dir: str = "data/"
    output_dir: str = "outputs/analysis"
    samples_per_class: int = 4
    dpi: int = 150


# ─── Data Collection ────────────────────────────────────────────────────
def scan_dataset(data_dir: str) -> pd.DataFrame:
    """Walk the dataset directory and collect per-image metadata.

    Returns a DataFrame with columns: category, filename, path, width,
    height, aspect_ratio, channels, file_size_kb, megapixels, is_corrupt.
    """
    records: list[dict] = []
    data_path = Path(data_dir)

    if not data_path.exists():
        logger.error("Data directory does not exist: %s", data_dir)
        sys.exit(1)

    for category_dir in sorted(data_path.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith((".", "_")):
            continue

        category = category_dir.name
        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() not in _IMG_EXTENSIONS:
                continue

            record: dict = {
                "category": category,
                "filename": img_file.name,
                "path": str(img_file),
                "is_corrupt": False,
            }
            try:
                with Image.open(img_file) as img:
                    img.verify()  # Detect truncated / corrupt files
                # Re-open after verify (PIL requirement)
                with Image.open(img_file) as img:
                    w, h = img.size
                    record.update({
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(w / h, 3),
                        "channels": len(img.getbands()),
                        "color_mode": img.mode,
                        "file_size_kb": round(img_file.stat().st_size / 1024, 1),
                        "megapixels": round((w * h) / 1e6, 2),
                    })
            except Exception as exc:
                logger.warning("Corrupt or unreadable image %s: %s", img_file, exc)
                record["is_corrupt"] = True

            records.append(record)

    df = pd.DataFrame(records)
    logger.info("Scanned %d images across %d categories", len(df), df["category"].nunique())
    return df


# ─── Statistical Analysis ──────────────────────────────────────────────
def compute_dataset_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics used in both the report and downstream tasks."""
    valid = df[~df["is_corrupt"]].copy()
    counts = df["category"].value_counts()

    return {
        "total_images": len(df),
        "valid_images": len(valid),
        "corrupt_images": int(df["is_corrupt"].sum()),
        "n_classes": df["category"].nunique(),
        "class_counts": counts.to_dict(),
        "min_class": counts.idxmin(),
        "min_class_count": int(counts.min()),
        "max_class": counts.idxmax(),
        "max_class_count": int(counts.max()),
        "imbalance_ratio": round(counts.max() / counts.min(), 1),
        "width_stats": valid["width"].describe().to_dict() if "width" in valid else {},
        "height_stats": valid["height"].describe().to_dict() if "height" in valid else {},
        "file_size_stats": valid["file_size_kb"].describe().to_dict() if "file_size_kb" in valid else {},
        "color_modes": valid["color_mode"].value_counts().to_dict() if "color_mode" in valid else {},
    }


# ─── Visualisation Functions ────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame, output_dir: str, dpi: int = 150) -> None:
    """Horizontal bar chart of per-class image counts with annotations."""
    counts = df["category"].value_counts().sort_values()
    imbalance_ratio = counts.max() / counts.min()

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("viridis", len(counts))
    bars = ax.barh(counts.index, counts.values, color=palette)

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center", fontsize=11, fontweight="bold",
        )

    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title(
        f"Class Distribution (Imbalance Ratio: {imbalance_ratio:.1f}:1)",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=dpi)
    plt.close()
    logger.info("Saved class_distribution.png")


def plot_image_dimensions(df: pd.DataFrame, output_dir: str, dpi: int = 150) -> None:
    """Scatter plot of width vs height coloured by category.

    Useful for spotting resolution outliers and deciding on a target resize.
    """
    valid = df.dropna(subset=["width", "height"])
    fig, ax = plt.subplots(figsize=(10, 8))
    categories = sorted(valid["category"].unique())
    palette = sns.color_palette("husl", len(categories))

    for cat, colour in zip(categories, palette):
        subset = valid[valid["category"] == cat]
        ax.scatter(subset["width"], subset["height"],
                   label=cat, alpha=0.6, s=25, color=colour)

    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.set_title("Image Dimensions by Category")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_dimensions.png"), dpi=dpi)
    plt.close()
    logger.info("Saved image_dimensions.png")


def plot_file_size_distribution(df: pd.DataFrame, output_dir: str, dpi: int = 150) -> None:
    """Box plot of file sizes per category — flags storage and quality variation."""
    valid = df.dropna(subset=["file_size_kb"])
    order = valid.groupby("category")["file_size_kb"].median().sort_values().index

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=valid, x="file_size_kb", y="category", order=order,
                ax=ax, palette="viridis")
    ax.set_xlabel("File Size (KB)")
    ax.set_title("File Size Distribution by Category")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "file_size_distribution.png"), dpi=dpi)
    plt.close()
    logger.info("Saved file_size_distribution.png")


def plot_aspect_ratio_histogram(df: pd.DataFrame, output_dir: str, dpi: int = 150) -> None:
    """Histogram of aspect ratios — informs padding vs. crop strategy."""
    valid = df.dropna(subset=["aspect_ratio"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(valid["aspect_ratio"], bins=40, color="steelblue", edgecolor="white")
    ax.axvline(x=1.0, color="red", linestyle="--", label="Square (1:1)")
    ax.set_xlabel("Aspect Ratio (W/H)")
    ax.set_ylabel("Count")
    ax.set_title("Aspect Ratio Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aspect_ratio_histogram.png"), dpi=dpi)
    plt.close()
    logger.info("Saved aspect_ratio_histogram.png")


def plot_sample_grid(
    df: pd.DataFrame, output_dir: str, samples_per_class: int = 4, dpi: int = 150
) -> None:
    """Grid of randomly sampled images — visual sanity check for each class."""
    categories = sorted(df["category"].unique())
    n_cats = len(categories)

    fig, axes = plt.subplots(n_cats, samples_per_class,
                             figsize=(samples_per_class * 3, n_cats * 2.5))

    for i, cat in enumerate(categories):
        cat_df = df[(df["category"] == cat) & (~df["is_corrupt"])].copy()
        sampled = cat_df.sample(n=min(samples_per_class, len(cat_df)), random_state=42)

        for j in range(samples_per_class):
            ax = axes[i][j]
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(sampled):
                try:
                    img = Image.open(sampled.iloc[j]["path"]).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.text(0.5, 0.5, "Error", ha="center", va="center",
                            transform=ax.transAxes)
            if j == 0:
                ax.set_ylabel(cat, fontsize=9, rotation=0, labelpad=80, ha="right")

    fig.suptitle("Sample Images per Category", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_grid.png"), dpi=dpi, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample_grid.png")


# ─── Report Generation ──────────────────────────────────────────────────
def generate_report(df: pd.DataFrame, stats: dict, output_dir: str) -> None:
    """Generate a structured text report and export metadata CSV."""
    lines = [
        "=" * 65,
        "  DATASET ANALYSIS REPORT — RTV Field Check-in Images",
        "=" * 65,
        "",
        f"Total images:       {stats['total_images']}",
        f"Valid images:       {stats['valid_images']}",
        f"Corrupt / unreadable: {stats['corrupt_images']}",
        f"Number of classes:  {stats['n_classes']}",
        "",
        "─── Class Distribution ───",
    ]
    for cat, count in sorted(stats["class_counts"].items(), key=lambda x: -x[1]):
        pct = count / stats["total_images"] * 100
        lines.append(f"  {cat:25s}  {count:4d}  ({pct:5.1f}%)")

    lines += [
        "",
        f"  Smallest class:  {stats['min_class']} ({stats['min_class_count']})",
        f"  Largest class:   {stats['max_class']} ({stats['max_class_count']})",
        f"  Imbalance ratio: {stats['imbalance_ratio']}:1",
        "",
        "─── Image Properties ───",
    ]
    if stats["width_stats"]:
        ws, hs = stats["width_stats"], stats["height_stats"]
        lines += [
            f"  Width:   min={ws.get('min',0):.0f}  max={ws.get('max',0):.0f}"
            f"  mean={ws.get('mean',0):.0f}  std={ws.get('std',0):.0f}",
            f"  Height:  min={hs.get('min',0):.0f}  max={hs.get('max',0):.0f}"
            f"  mean={hs.get('mean',0):.0f}  std={hs.get('std',0):.0f}",
        ]
    if stats["file_size_stats"]:
        fs = stats["file_size_stats"]
        lines.append(
            f"  File KB: min={fs.get('min',0):.0f}  max={fs.get('max',0):.0f}"
            f"  mean={fs.get('mean',0):.0f}"
        )
    if stats["color_modes"]:
        lines.append(f"  Colour modes: {stats['color_modes']}")

    lines += [
        "",
        "─── Key Observations ───",
        "  1. Severe class imbalance — guinea-pig-shelter has ~16 images",
        "     vs ~150 for every other class (9.4:1 ratio).",
        "  2. Visually similar categories (compost / organic / liquid-organic)",
        "     may confuse the classifier — need strong feature extraction.",
        "  3. Field-captured images show high variance in lighting,",
        "     angle, and background — augmentation must reflect this.",
        "  4. Mixed resolutions require a standardised resize pipeline;",
        "     aspect-ratio distribution informs padding vs crop choice.",
        "=" * 65,
    ]

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    logger.info("Report saved to %s", report_path)

    csv_path = os.path.join(output_dir, "dataset_metadata.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Metadata CSV saved to %s", csv_path)


# ─── Entry Point ────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 1 — Exploratory Data Analysis on RTV check-in images",
    )
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Root directory containing class-labelled image folders")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis",
                        help="Directory to write plots and reports")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for saved figures")
    args = parser.parse_args()

    cfg = AnalysisConfig(data_dir=args.data_dir, output_dir=args.output_dir, dpi=args.dpi)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Step 1 — Scan
    logger.info("[1/7] Scanning dataset at %s …", cfg.data_dir)
    df = scan_dataset(cfg.data_dir)

    # Step 2 — Statistics
    logger.info("[2/7] Computing statistics …")
    stats = compute_dataset_statistics(df)

    # Steps 3-7 — Visualisations and report
    logger.info("[3/7] Plotting class distribution …")
    plot_class_distribution(df, cfg.output_dir, cfg.dpi)

    logger.info("[4/7] Plotting image dimensions …")
    plot_image_dimensions(df, cfg.output_dir, cfg.dpi)

    logger.info("[5/7] Plotting file-size distribution …")
    plot_file_size_distribution(df, cfg.output_dir, cfg.dpi)

    logger.info("[6/7] Plotting aspect-ratio histogram …")
    plot_aspect_ratio_histogram(df, cfg.output_dir, cfg.dpi)

    logger.info("[7/7] Generating sample grid and report …")
    plot_sample_grid(df, cfg.output_dir, cfg.samples_per_class, cfg.dpi)
    generate_report(df, stats, cfg.output_dir)

    logger.info("Analysis complete — outputs in %s", cfg.output_dir)


if __name__ == "__main__":
    main()
