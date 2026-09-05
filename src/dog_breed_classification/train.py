"""Training pipeline for Dog Breed Classification.

Implements a two-phase transfer learning workflow:
- Phase 1: Feature extraction (training custom classification head with frozen backbone).
- Phase 2: Fine-tuning (training upper layers with a lower learning rate).
Includes comprehensive callbacks (Checkpoints, EarlyStopping, TensorBoard, CSVLogger, LR Schedule),
Apple Silicon GPU optimization, and training curve visualization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from dog_breed_classification.config import (
    TrainingConfig,
    setup_device,
)
from dog_breed_classification.dataset import (
    build_tf_dataset,
    create_stratified_splits,
    get_class_mappings,
    load_dataset_index,
)
from dog_breed_classification.models import (
    build_dog_classifier,
    compile_model,
    set_backbone_trainable,
)


def get_callbacks(
    config: TrainingConfig,
    phase: str = "phase1",
) -> List[tf.keras.callbacks.Callback]:
    """Builds standard Keras callbacks for training stability and artifact logging.

    Args:
        config: TrainingConfig instance.
        phase: Current training phase ('phase1' or 'phase2').

    Returns:
        List of configured tf.keras.callbacks.Callback objects.
    """
    config.ensure_directories()
    checkpoint_path = (
        config.checkpoints_dir / f"{config.model_name}_{phase}_best.weights.h5"
    )
    log_dir = config.logs_dir / f"{config.model_name}_{phase}_{int(time.time())}"
    csv_path = config.logs_dir / f"{config.model_name}_{phase}_history.csv"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=5 if phase == "phase1" else 6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(csv_path),
            separator=",",
            append=False,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=0,
            write_graph=False,
        ),
    ]
    return callbacks


def plot_training_history(
    history_dict: Dict[str, List[float]],
    save_path: Optional[Path] = None,
    show: bool = False,
) -> plt.Figure:
    """Plots training and validation accuracy/loss curves across all phases.

    Args:
        history_dict: Combined metrics dictionary from training phases.
        save_path: Destination file path for figure.
        show: Whether to display figure interactively.

    Returns:
        matplotlib.figure.Figure instance.
    """
    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    ax1.plot(epochs, history_dict.get("accuracy", []), "b-o", label="Training Accuracy")
    if "val_accuracy" in history_dict:
        ax1.plot(
            epochs,
            history_dict.get("val_accuracy", []),
            "r--s",
            label="Validation Accuracy",
        )
    if "top_5_accuracy" in history_dict:
        ax1.plot(
            epochs,
            history_dict.get("top_5_accuracy", []),
            "g-^",
            label="Train Top-5 Acc",
        )
    if "val_top_5_accuracy" in history_dict:
        ax1.plot(
            epochs,
            history_dict.get("val_top_5_accuracy", []),
            "m--v",
            label="Val Top-5 Acc",
        )

    ax1.set_title("Model Accuracy over Epochs", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Loss Plot
    ax2.plot(epochs, history_dict.get("loss", []), "b-o", label="Training Loss")
    if "val_loss" in history_dict:
        ax2.plot(epochs, history_dict.get("val_loss", []), "r--s", label="Validation Loss")

    ax2.set_title("Model Loss over Epochs", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss (Categorical Crossentropy)", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def merge_histories(
    hist1: Dict[str, List[float]],
    hist2: Dict[str, List[float]],
) -> Dict[str, List[float]]:
    """Concatenates metric histories from Phase 1 and Phase 2."""
    merged = {}
    all_keys = set(hist1.keys()).union(set(hist2.keys()))
    for k in all_keys:
        merged[k] = hist1.get(k, []) + hist2.get(k, [])
    return merged


def train_model(
    config: Optional[TrainingConfig] = None,
    df: Optional[pd.DataFrame] = None,
    verbose: int = 1,
) -> Tuple[tf.keras.Model, Dict[str, List[float]], Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Executes the complete dog breed training pipeline.

    Args:
        config: TrainingConfig object (defaults to standard config).
        df: Optional pre-loaded dataset DataFrame.
        verbose: Keras verbosity level (1=progress bar, 2=one line per epoch).

    Returns:
        Tuple of (trained_model, combined_history_dict, (train_df, val_df, test_df)).
    """
    if config is None:
        config = TrainingConfig()

    config.ensure_directories()
    device_type = setup_device(verbose=bool(verbose))
    if verbose:
        print(f"[*] Starting Dog Breed Classification Training [{config.model_name}]")
        print(f"[*] Compute Target: {device_type}")

    # 1. Dataset Loading & Class Mapping
    if df is None:
        df = load_dataset_index()

    class_mapping_path = config.models_dir / "class_names.json"
    class_to_idx, idx_to_class, class_names = get_class_mappings(
        df,
        save_path=class_mapping_path,
    )
    if verbose:
        print(
            f"[*] Indexed {len(df)} images across {len(class_names)} dog breeds. "
            f"Class mapping saved to {class_mapping_path}"
        )

    # 2. Stratified Train / Val / Test Splits
    train_df, val_df, test_df = create_stratified_splits(
        df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    if verbose:
        print(
            f"[*] Data Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

    # 3. tf.data Datasets
    train_ds = build_tf_dataset(
        train_df,
        class_to_idx=class_to_idx,
        image_size=config.image_size,
        batch_size=config.batch_size,
        is_training=True,
        augment=config.use_augmentation,
        num_classes=len(class_names),
    )
    val_ds = build_tf_dataset(
        val_df,
        class_to_idx=class_to_idx,
        image_size=config.image_size,
        batch_size=config.batch_size,
        is_training=False,
        augment=False,
        num_classes=len(class_names),
    )

    # 4. Model Construction (Phase 1: Feature Extraction)
    if verbose:
        print(f"\n{'='*60}\n[Phase 1] Feature Extraction (Backbone Frozen)\n{'='*60}")

    model = build_dog_classifier(
        model_name=config.model_name,
        input_shape=(config.image_size[0], config.image_size[1], 3),
        num_classes=len(class_names),
        dropout_rate=config.dropout_rate,
        l2_reg=config.l2_reg,
        freeze_backbone=True,
    )
    model = compile_model(
        model,
        learning_rate=config.initial_lr,
        label_smoothing=config.label_smoothing,
    )

    phase1_callbacks = get_callbacks(config, phase="phase1")
    history_p1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.initial_epochs,
        callbacks=phase1_callbacks,
        verbose=verbose,
    )
    combined_history = history_p1.history

    # 5. Phase 2: Fine-Tuning
    if config.fine_tune and config.fine_tune_epochs > 0:
        if verbose:
            print(
                f"\n{'='*60}\n[Phase 2] Fine-Tuning (Unfreezing top {config.fine_tune_layers} layers)\n{'='*60}"
            )

        model = set_backbone_trainable(
            model,
            unfreeze_layers=config.fine_tune_layers,
            learning_rate=config.fine_tune_lr,
            label_smoothing=config.label_smoothing,
        )

        phase2_callbacks = get_callbacks(config, phase="phase2")
        history_p2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.fine_tune_epochs,
            callbacks=phase2_callbacks,
            verbose=verbose,
        )
        combined_history = merge_histories(history_p1.history, history_p2.history)

    # 6. Save Model & History Artifacts
    final_model_path = config.models_dir / f"dog_classifier_{config.model_name}.keras"
    model.save(str(final_model_path))
    if verbose:
        print(f"\n[*] Model saved to: {final_model_path}")

    # Save metrics JSON
    history_json_path = (
        config.logs_dir / f"{config.model_name}_training_history.json"
    )
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: [float(v) for v in vals] for k, vals in combined_history.items()},
            f,
            indent=2,
        )

    # Save training curve plots
    plot_path = config.plots_dir / f"{config.model_name}_training_curves.png"
    plot_training_history(combined_history, save_path=plot_path)
    if verbose:
        print(f"[*] Training curves plot saved to: {plot_path}")

    return model, combined_history, (train_df, val_df, test_df)
