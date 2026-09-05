"""Evaluation metrics, diagnostic reports, and confusion visualizers for Dog Breed Classification.

Computes:
- Top-1 and Top-5 Categorical Accuracy.
- Precision, Recall, F1-Score (macro, weighted, and per-class).
- Confusion matrix heatmaps and Top-K confused breed pairs analysis.
- Qualitative sample prediction galleries with visual confidence overlays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

from dog_breed_classification.config import (
    DEFAULT_IMAGE_SIZE,
    PLOTS_DIR,
    REPORTS_DIR,
)
from dog_breed_classification.dataset import build_tf_dataset


def evaluate_model(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    class_names: List[str],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = 32,
    save_reports: bool = True,
    reports_dir: Path = REPORTS_DIR,
    plots_dir: Path = PLOTS_DIR,
) -> Dict[str, Union[float, Dict, List]]:
    """Evaluates a trained dog breed classifier on test data.

    Args:
        model: Trained tf.keras.Model.
        test_df: Test dataset DataFrame.
        class_to_idx: Class name to integer index mapping.
        class_names: Ordered list of class names.
        image_size: Target image dimensions.
        batch_size: Batch size for inference.
        save_reports: Whether to save report and visualization files.
        reports_dir: Directory for JSON and CSV reports.
        plots_dir: Directory for generated charts.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    reports_dir = Path(reports_dir)
    plots_dir = Path(plots_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    test_ds = build_tf_dataset(
        test_df,
        class_to_idx=class_to_idx,
        image_size=image_size,
        batch_size=batch_size,
        is_training=False,
        augment=False,
        num_classes=len(class_names),
    )

    y_true_indices = np.array([class_to_idx[b] for b in test_df["breed"]])

    # Model Predictions
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # Core Metrics
    top_1_acc = float(accuracy_score(y_true_indices, y_pred_indices))
    try:
        top_5_acc = float(
            top_k_accuracy_score(
                y_true_indices,
                y_pred_probs,
                k=5,
                labels=np.arange(len(class_names)),
            )
        )
    except Exception:
        top_5_acc = None

    macro_precision = float(
        precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0)
    )
    macro_f1 = float(
        f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0)
    )
    weighted_f1 = float(
        f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0)
    )

    # Classification Report
    report_dict = classification_report(
        y_true_indices,
        y_pred_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # Top Confused Pairs
    conf_matrix = confusion_matrix(
        y_true_indices, y_pred_indices, labels=np.arange(len(class_names))
    )
    np.fill_diagonal(conf_matrix, 0)  # zero out correct classifications

    confused_pairs = []
    for true_idx in range(len(class_names)):
        for pred_idx in range(len(class_names)):
            count = int(conf_matrix[true_idx, pred_idx])
            if count > 0:
                confused_pairs.append(
                    {
                        "true_breed": class_names[true_idx],
                        "predicted_breed": class_names[pred_idx],
                        "confusion_count": count,
                    }
                )
    confused_pairs = sorted(
        confused_pairs, key=lambda x: x["confusion_count"], reverse=True
    )

    results = {
        "total_test_samples": len(test_df),
        "num_classes": len(class_names),
        "top_1_accuracy": top_1_acc,
        "top_5_accuracy": top_5_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "top_10_confusions": confused_pairs[:10],
    }

    if save_reports:
        # Save JSON Report
        report_file = reports_dir / "evaluation_summary.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Save Per-Class CSV Report
        per_class_df = pd.DataFrame(report_dict).transpose()
        per_class_df.to_csv(reports_dir / "classification_report.csv")

        # Save Visualizations
        plot_top_confusions(
            confused_pairs,
            top_k=12,
            save_path=plots_dir / "top_confused_breeds.png",
        )
        plot_accuracy_distribution(
            report_dict,
            class_names,
            top_k=15,
            save_path=plots_dir / "breed_accuracy_extremes.png",
        )

    return results


def plot_top_confusions(
    confused_pairs: List[Dict],
    top_k: int = 12,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plots a bar chart of the most frequently confused breed pairs."""
    top_items = confused_pairs[:top_k]
    if not top_items:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No classification errors detected.", ha="center")
        return fig

    labels = [
        f"{item['true_breed']}\n-> {item['predicted_breed']}" for item in top_items
    ]
    counts = [item["confusion_count"] for item in top_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds_r(np.linspace(0.2, 0.7, len(top_items)))
    bars = ax.barh(range(len(top_items)), counts, color=colors, edgecolor="black")
    ax.set_yticks(range(len(top_items)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Misclassifications", fontsize=12)
    ax.set_title(
        f"Top {len(top_items)} Most Confused Dog Breed Pairs",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_accuracy_distribution(
    report_dict: Dict,
    class_names: List[str],
    top_k: int = 15,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plots top-K highest and lowest performing dog breeds by F1-Score."""
    class_scores = []
    for name in class_names:
        if name in report_dict and isinstance(report_dict[name], dict):
            class_scores.append((name, report_dict[name].get("f1-score", 0.0)))

    class_scores.sort(key=lambda x: x[1])
    lowest = class_scores[:top_k]
    highest = class_scores[-top_k:][::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Highest performing
    names_h, scores_h = zip(*highest) if highest else ([], [])
    y_pos_h = range(len(names_h))
    ax1.barh(y_pos_h, scores_h, color="#2ca02c", edgecolor="black")
    ax1.set_yticks(y_pos_h)
    ax1.set_yticklabels(names_h, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("F1-Score", fontsize=12)
    ax1.set_title(f"Top {len(names_h)} Highest Performing Breeds", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    # Lowest performing
    names_l, scores_l = zip(*lowest) if lowest else ([], [])
    y_pos_l = range(len(names_l))
    ax2.barh(y_pos_l, scores_l, color="#d62728", edgecolor="black")
    ax2.set_yticks(y_pos_l)
    ax2.set_yticklabels(names_l, fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("F1-Score", fontsize=12)
    ax2.set_title(f"Top {len(names_l)} Most Challenging Breeds", fontsize=13, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_sample_predictions_grid(
    model: tf.keras.Model,
    test_df: pd.DataFrame,
    class_names: List[str],
    class_to_idx: Dict[str, int],
    num_samples: int = 12,
    num_cols: int = 4,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Generates a visual grid showing dog test images with true vs predicted labels
    and confidence scores.
    """
    sample_df = test_df.sample(n=min(num_samples, len(test_df)), random_state=42).reset_index(drop=True)
    num_rows = int(np.ceil(len(sample_df) / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 3.8))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, row in sample_df.iterrows():
        ax = axes[idx]
        img_path = row["filepath"]
        true_breed = row["breed"]

        # Load raw image
        raw_img = Image.open(img_path).convert("RGB")
        img_resized = raw_img.resize(image_size)
        img_arr = np.array(img_resized, dtype=np.float32)
        batch_input = np.expand_dims(img_arr, axis=0)

        # Inference
        probs = model.predict(batch_input, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_breed = class_names[pred_idx]
        confidence = float(probs[pred_idx])

        is_correct = (pred_breed == true_breed)
        color = "#1b8a34" if is_correct else "#c82333"

        ax.imshow(raw_img)
        ax.set_title(
            f"True: {true_breed}\nPred: {pred_breed} ({confidence*100:.1f}%)",
            fontsize=9.5,
            color=color,
            fontweight="bold",
        )
        ax.axis("off")

        # Visual border for correctness
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Hide unused subplots
    for idx in range(len(sample_df), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
