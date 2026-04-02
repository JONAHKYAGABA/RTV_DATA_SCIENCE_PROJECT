"""
Task 2: Model Evaluation
=========================
Rigorous evaluation of the trained YOLO classification model, producing:
  - Per-class precision / recall / F1-score
  - Macro and weighted F1 (macro is preferred for imbalanced datasets
    because it weights every class equally, exposing minority-class weakness)
  - Confusion matrix (raw + normalised)
  - Confidence calibration analysis
  - Misclassification audit with error taxonomy
  - All metrics persisted as JSON for downstream comparison

Usage:
    python src/task2_model_evaluation.py \
        --model outputs/training/<run>/weights/best.pt \
        --data_dir outputs/prepared_data
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# ─── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_IMG_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Inference ──────────────────────────────────────────────────────────
def load_model(model_path: str):
    """Load a trained YOLO model from a checkpoint."""
    from ultralytics import YOLO
    model = YOLO(model_path)
    logger.info("Model loaded from %s", model_path)
    return model


def run_inference(
    model, data_dir: str, split: str = "test", imgsz: int = 640,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Classify every image in the given split directory.

    Returns:
        results: list of per-image prediction dicts
        class_names: sorted list of ground-truth class names
    """
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    class_dirs = sorted(d for d in split_dir.iterdir() if d.is_dir())
    class_names = [d.name for d in class_dirs]
    results: list[dict[str, Any]] = []

    from tqdm import tqdm

    for class_dir in class_dirs:
        true_label = class_dir.name
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in _IMG_EXTENSIONS]

        for img_path in tqdm(images, desc=f"  {true_label}", leave=False):
            try:
                preds = model.predict(str(img_path), imgsz=imgsz, verbose=False)
                pred = preds[0]

                top1_idx = pred.probs.top1
                pred_label = pred.names[top1_idx]
                top1_conf = float(pred.probs.top1conf)
                top5_indices = pred.probs.top5

                results.append({
                    "path": str(img_path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": top1_conf,
                    "correct": true_label == pred_label,
                    "top5_labels": [pred.names[i] for i in top5_indices],
                    "top5_confs": [float(pred.probs.data[i]) for i in top5_indices],
                    "in_top5": true_label in [pred.names[i] for i in top5_indices],
                })
            except Exception as exc:
                logger.warning("Inference failed on %s: %s", img_path, exc)

    logger.info("Inference complete: %d images, %d classes", len(results), len(class_names))
    return results, class_names


# ─── Metrics ────────────────────────────────────────────────────────────
def compute_metrics(results: list[dict], class_names: list[str]) -> dict[str, Any]:
    """Compute a comprehensive metrics dictionary from predictions."""
    y_true = [r["true_label"] for r in results]
    y_pred = [r["pred_label"] for r in results]
    confs = [r["confidence"] for r in results]
    correct_confs = [r["confidence"] for r in results if r["correct"]]
    wrong_confs = [r["confidence"] for r in results if not r["correct"]]

    report = classification_report(
        y_true, y_pred, labels=class_names, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    return {
        "top1_accuracy": sum(r["correct"] for r in results) / len(results),
        "top5_accuracy": sum(r["in_top5"] for r in results) / len(results),
        "macro_f1": f1_score(y_true, y_pred, labels=class_names,
                             average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=class_names,
                                average="weighted", zero_division=0),
        "total_samples": len(results),
        "correct": sum(r["correct"] for r in results),
        "incorrect": sum(not r["correct"] for r in results),
        "mean_confidence": float(np.mean(confs)),
        "mean_conf_correct": float(np.mean(correct_confs)) if correct_confs else 0.0,
        "mean_conf_wrong": float(np.mean(wrong_confs)) if wrong_confs else 0.0,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


def print_metrics(m: dict[str, Any]) -> None:
    """Pretty-print the most important metrics to the console."""
    logger.info("=" * 65)
    logger.info("  EVALUATION RESULTS")
    logger.info("=" * 65)
    logger.info("  Samples     : %d", m["total_samples"])
    logger.info("  Top-1 Acc   : %.4f  (%d / %d)", m["top1_accuracy"], m["correct"], m["total_samples"])
    logger.info("  Top-5 Acc   : %.4f", m["top5_accuracy"])
    logger.info("  Macro F1    : %.4f  (treats every class equally)", m["macro_f1"])
    logger.info("  Weighted F1 : %.4f  (accounts for class size)", m["weighted_f1"])
    logger.info("  Conf (ok)   : %.4f  |  Conf (err) : %.4f",
                m["mean_conf_correct"], m["mean_conf_wrong"])
    logger.info("")
    header = f"  {'Class':25s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}"
    logger.info(header)
    logger.info("  " + "-" * 57)
    for cls in m["class_names"]:
        r = m["per_class"].get(cls, {})
        logger.info("  %-25s %8.3f %8.3f %8.3f %8.0f",
                     cls, r.get("precision", 0), r.get("recall", 0),
                     r.get("f1-score", 0), r.get("support", 0))


# ─── Visualisations ────────────────────────────────────────────────────
def plot_confusion_matrix(m: dict, output_dir: str) -> None:
    """Side-by-side raw and normalised confusion matrices."""
    cm = np.array(m["confusion_matrix"])
    names = m["class_names"]
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    for ax, data, fmt, title in [
        (axes[0], cm, "d", "Confusion Matrix (Counts)"),
        (axes[1], cm_norm, ".2f", "Confusion Matrix (Normalised)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved confusion_matrix.png")


def plot_confidence_histogram(results: list[dict], output_dir: str) -> None:
    """Confidence distributions for correct vs. incorrect predictions.

    A well-calibrated model should show high confidence for correct
    predictions and lower, spread confidence for incorrect ones.
    """
    ok = [r["confidence"] for r in results if r["correct"]]
    err = [r["confidence"] for r in results if not r["correct"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 30)
    ax.hist(ok, bins=bins, alpha=0.7, color="green", edgecolor="darkgreen",
            label=f"Correct (n={len(ok)})")
    if err:
        ax.hist(err, bins=bins, alpha=0.7, color="red", edgecolor="darkred",
                label=f"Incorrect (n={len(err)})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_histogram.png"), dpi=150)
    plt.close()
    logger.info("Saved confidence_histogram.png")


def plot_per_class_f1(m: dict, output_dir: str) -> None:
    """Horizontal bar chart of per-class F1 scores."""
    names = m["class_names"]
    scores = [m["per_class"].get(c, {}).get("f1-score", 0) for c in names]
    colours = ["#d32f2f" if s < 0.5 else "#ff9800" if s < 0.7 else "#388e3c" for s in scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, scores, color=colours, edgecolor="grey")
    for bar, val in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)
    ax.axvline(m["macro_f1"], color="blue", ls="--", alpha=0.5,
               label=f"Macro F1 = {m['macro_f1']:.3f}")
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Class F1 Scores")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=150)
    plt.close()
    logger.info("Saved per_class_f1.png")


# ─── Error Analysis ────────────────────────────────────────────────────
def misclassification_audit(results: list[dict], output_dir: str) -> None:
    """Detailed breakdown of where and why the model fails.

    Outputs:
      - Most common confusion pairs
      - Low-confidence correct predictions (likely to flip with noise)
      - High-confidence incorrect predictions (most dangerous in production)
    """
    errors = [r for r in results if not r["correct"]]
    if not errors:
        logger.info("No misclassifications — perfect score on this split!")
        return

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for r in errors:
        pair_counts[(r["true_label"], r["pred_label"])] += 1

    sorted_pairs = sorted(pair_counts.items(), key=lambda x: -x[1])

    lines = [
        "MISCLASSIFICATION AUDIT",
        "=" * 65,
        f"Total errors: {len(errors)} / {len(results)} ({len(errors)/len(results)*100:.1f}%)",
        "",
        "Top confusion pairs (true -> predicted : count):",
    ]
    for (true, pred), cnt in sorted_pairs[:10]:
        lines.append(f"  {true:25s} -> {pred:25s} : {cnt}")

    # High-confidence errors are the most dangerous in production
    high_conf_errors = sorted(errors, key=lambda r: -r["confidence"])[:5]
    lines += ["", "Highest-confidence errors (most dangerous):"]
    for r in high_conf_errors:
        lines.append(
            f"  {r['true_label']:25s} predicted as {r['pred_label']:25s}"
            f"  conf={r['confidence']:.3f}  ({Path(r['path']).name})"
        )

    # Low-confidence correct predictions — fragile
    fragile = sorted([r for r in results if r["correct"]], key=lambda r: r["confidence"])[:5]
    lines += ["", "Lowest-confidence correct predictions (fragile):"]
    for r in fragile:
        lines.append(f"  {r['true_label']:25s}  conf={r['confidence']:.3f}")

    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(output_dir, "misclassification_audit.txt"), "w") as fh:
        fh.write(report)
    logger.info("Saved misclassification_audit.txt")


# ─── Entry Point ────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2 — Model Evaluation")
    parser.add_argument("--model", required=True, help="Path to best.pt")
    parser.add_argument("--data_dir", default="outputs/prepared_data")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output_dir", default="outputs/evaluation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.model)
    results, class_names = run_inference(model, args.data_dir, args.split, args.imgsz)

    metrics = compute_metrics(results, class_names)
    print_metrics(metrics)

    # Plots
    plot_confusion_matrix(metrics, args.output_dir)
    plot_confidence_histogram(results, args.output_dir)
    plot_per_class_f1(metrics, args.output_dir)
    misclassification_audit(results, args.output_dir)

    # Persist
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    with open(os.path.join(args.output_dir, "predictions.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    logger.info("Evaluation complete — outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
