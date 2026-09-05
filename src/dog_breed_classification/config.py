"""Configuration module for Dog Breed Classification.

Centralizes paths, hyperparameters, dataset specifications, and hardware settings
with optimizations for Apple Silicon (M-series) GPUs via Metal and CPU fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

# Base Project Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images" / "Images"
ANNOTATIONS_DIR = DATA_DIR / "annotations" / "Annotation"

# Output and Artifact Directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ARTIFACTS_DIR / "logs"
PLOTS_DIR = ARTIFACTS_DIR / "plots"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

# Dataset Specs
NUM_CLASSES = 120
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_CHANNELS: int = 3
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_SEED: int = 42

# Train / Val / Test Splits
DEFAULT_TRAIN_RATIO: float = 0.70
DEFAULT_VAL_RATIO: float = 0.15
DEFAULT_TEST_RATIO: float = 0.15

# Supported Pretrained Backbones
SUPPORTED_BACKBONES = [
    "efficientnetv2_s",
    "efficientnetv2_b0",
    "mobilenetv3_large",
    "resnet50v2",
]


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration settings for model training."""

    model_name: str = "efficientnetv2_s"
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_classes: int = 120
    seed: int = 42

    # Training Phase 1: Feature Extraction (Backbone Frozen)
    initial_epochs: int = 10
    initial_lr: float = 1e-3

    # Training Phase 2: Fine-Tuning (Upper Backbone Unfrozen)
    fine_tune: bool = True
    fine_tune_epochs: int = 15
    fine_tune_lr: float = 1e-4
    fine_tune_layers: int = 50  # Unfreeze top N layers of backbone

    # Regularization & Optimization
    dropout_rate: float = 0.3
    label_smoothing: float = 0.1
    l2_reg: float = 1e-4
    use_augmentation: bool = True

    # Data split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Storage paths
    models_dir: Path = MODELS_DIR
    checkpoints_dir: Path = CHECKPOINTS_DIR
    logs_dir: Path = LOGS_DIR
    plots_dir: Path = PLOTS_DIR
    reports_dir: Path = REPORTS_DIR

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for directory in [
            self.models_dir,
            self.checkpoints_dir,
            self.logs_dir,
            self.plots_dir,
            self.reports_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def setup_device(verbose: bool = True) -> str:
    """Configures TensorFlow device for macOS Apple Silicon (Metal GPU) or CPU.

    Returns:
        Device type string ("GPU" or "CPU").
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            device_type = "GPU"
            if verbose:
                print(f"[Device] Utilizing Apple Silicon Metal GPU: {gpus}")
        except Exception as e:
            device_type = "GPU (Memory growth configuration warning: " + str(e) + ")"
            if verbose:
                print(f"[Device] GPU detected with note: {e}")
    else:
        device_type = "CPU"
        if verbose:
            print("[Device] No GPU detected. Utilizing CPU execution.")

    return device_type
