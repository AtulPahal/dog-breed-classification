"""Dataset loading, indexing, preprocessing, and tf.data pipeline for Stanford Dogs.

Provides utilities for:
- Standardizing and cleaning raw ImageNet breed folder names.
- Extracting metadata and optional bounding boxes from annotations.
- Stratified train/val/test splitting across 120 breeds.
- Building high-performance tf.data pipelines with GPU-accelerated augmentations.
"""

from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dog_breed_classification.config import (
    ANNOTATIONS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_SEED,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    IMAGES_DIR,
    MODELS_DIR,
    NUM_CLASSES,
)


def clean_breed_name(raw_name: str) -> str:
    """Transforms raw Stanford Dogs folder names (e.g. 'n02085620-Chihuahua',
    'n02099601-golden_retriever') into clean, capitalized display names.

    Args:
        raw_name: Raw folder or identifier name.

    Returns:
        Formatted breed string (e.g. 'Golden Retriever', 'Chihuahua').
    """
    # Strip ImageNet ID prefix (e.g., 'n02085620-')
    cleaned = re.sub(r"^n\d+-", "", raw_name)
    # Replace underscores and hyphens with spaces (keeping special hyphenated breeds if needed)
    words = cleaned.replace("_", " ").split()
    formatted_words = []
    for word in words:
        # Handle cases like Shih-Tzu or English Springer
        if "-" in word:
            parts = word.split("-")
            formatted_words.append("-".join(p.capitalize() for p in parts))
        else:
            formatted_words.append(word.capitalize())
    return " ".join(formatted_words)


def parse_annotation_xml(xml_path: Union[str, Path]) -> Optional[Dict]:
    """Parses Pascal VOC XML annotation for a dog image.

    Args:
        xml_path: Path to annotation XML file.

    Returns:
        Dictionary with image size and list of bounding boxes, or None on failure.
    """
    path = Path(xml_path)
    if not path.is_file():
        return None

    try:
        tree = ET.parse(path)
        root = tree.getroot()
        size_elem = root.find("size")
        width = int(size_elem.find("width").text) if size_elem is not None else None
        height = int(size_elem.find("height").text) if size_elem is not None else None
        depth = int(size_elem.find("depth").text) if size_elem is not None else 3

        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bnd = obj.find("bndbox")
            if bnd is not None:
                box = {
                    "xmin": int(bnd.find("xmin").text),
                    "ymin": int(bnd.find("ymin").text),
                    "xmax": int(bnd.find("xmax").text),
                    "ymax": int(bnd.find("ymax").text),
                }
                objects.append({"name": name, "bndbox": box})

        return {"width": width, "height": height, "depth": depth, "objects": objects}
    except Exception:
        return None


def load_dataset_index(
    images_dir: Union[str, Path] = IMAGES_DIR,
    annotations_dir: Optional[Union[str, Path]] = ANNOTATIONS_DIR,
) -> pd.DataFrame:
    """Scans dataset directory, indexes all images, extracts class labels,
    and returns a structured pandas DataFrame.

    Args:
        images_dir: Directory containing class subfolders of dog images.
        annotations_dir: Optional directory with Pascal VOC annotation files.

    Returns:
        pd.DataFrame containing columns:
        ['filepath', 'filename', 'raw_class', 'breed', 'class_id', 'annotation_path']
    """
    images_path = Path(images_dir)
    annotations_path = Path(annotations_dir) if annotations_dir else None

    if not images_path.exists():
        raise FileNotFoundError(
            f"Images directory not found at: {images_path}. "
            "Please ensure dataset is located in data/images/Images or specify custom path."
        )

    records = []
    class_folders = sorted(
        [d for d in images_path.iterdir() if d.is_dir() and not d.name.startswith(".")]
    )

    for class_id, class_folder in enumerate(class_folders):
        raw_class = class_folder.name
        breed = clean_breed_name(raw_class)

        for img_file in class_folder.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in (
                ".jpg",
                ".jpeg",
                ".png",
            ):
                stem = img_file.stem
                anno_file = (
                    (annotations_path / raw_class / stem)
                    if annotations_path and (annotations_path / raw_class / stem).exists()
                    else None
                )

                records.append(
                    {
                        "filepath": str(img_file.resolve()),
                        "filename": img_file.name,
                        "raw_class": raw_class,
                        "breed": breed,
                        "class_id": class_id,
                        "annotation_path": str(anno_file) if anno_file else None,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"No valid dog image files found in {images_path}")

    return df


def get_class_mappings(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """Generates class mappings from dataset DataFrame.

    Args:
        df: Dataset DataFrame.
        save_path: Optional path to save class mapping JSON.

    Returns:
        Tuple of (class_to_idx, idx_to_class, class_names_list).
    """
    unique_breeds = sorted(df["breed"].unique())
    class_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
    idx_to_class = {idx: breed for idx, breed in enumerate(unique_breeds)}
    class_names = [idx_to_class[i] for i in range(len(unique_breeds))]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_classes": len(class_names),
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            "class_names": class_names,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return class_to_idx, idx_to_class, class_names


def load_class_mappings(
    json_path: Union[str, Path],
) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """Loads class mappings from a previously saved JSON file.

    Args:
        json_path: Path to class_names.json.

    Returns:
        Tuple of (class_to_idx, idx_to_class, class_names_list).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_to_idx = data["class_to_idx"]
    idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
    class_names = data["class_names"]
    return class_to_idx, idx_to_class, class_names


def create_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits dataset into stratified train, validation, and test subsets.

    Args:
        df: Dataset DataFrame.
        train_ratio: Proportion of training data (e.g. 0.70).
        val_ratio: Proportion of validation data (e.g. 0.15).
        test_ratio: Proportion of testing data (e.g. 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    total = train_ratio + val_ratio + test_ratio
    train_r = train_ratio / total
    val_r = val_ratio / total
    test_r = test_ratio / total
    # Check if stratified split is possible (each class needs at least 2 samples and test_size >= n_classes)
    class_counts = df["breed"].value_counts()
    min_count = class_counts.min()
    n_classes = len(class_counts)
    can_stratify = (min_count >= 2) and (int(len(df) * (1.0 - train_r)) >= n_classes)

    # Step 1: Separate Train and (Val + Test)
    train_df, val_test_df = train_test_split(
        df,
        train_size=train_r,
        stratify=df["breed"] if can_stratify else None,
        random_state=seed,
        shuffle=True,
    )

    # Step 2: Separate Val and Test from remaining data
    val_rel_ratio = val_r / (val_r + test_r)
    val_can_stratify = can_stratify and (
        int(len(val_test_df) * (1.0 - val_rel_ratio)) >= val_test_df["breed"].nunique()
    )
    val_df, test_df = train_test_split(
        val_test_df,
        train_size=val_rel_ratio,
        stratify=val_test_df["breed"] if val_can_stratify else None,
        random_state=seed,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )

def create_augmentation_pipeline(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
):
    """Creates a TensorFlow Keras data augmentation sequential model.

    Args:
        image_size: Target image dimensions (height, width).

    Returns:
        tf.keras.Sequential augmentation layer pipeline.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.12, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.12, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_tf_dataset(
    df: pd.DataFrame,
    class_to_idx: Dict[str, int],
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    is_training: bool = True,
    augment: bool = True,
    num_classes: int = NUM_CLASSES,
    cache: bool = False,
    shuffle_buffer: int = 1024,
):
    """Builds a high-performance tf.data.Dataset pipeline.

    Args:
        df: DataFrame containing 'filepath' and 'breed'.
        class_to_idx: Dictionary mapping breed name to numerical index.
        image_size: Target image size (height, width).
        batch_size: Mini-batch size.
        is_training: Whether dataset is used for training (enables shuffle).
        augment: Whether to apply data augmentation.
        num_classes: Total number of classes (120 for Stanford Dogs).
        cache: Whether to cache in memory.
        shuffle_buffer: Buffer size for random shuffling.

    Returns:
        tf.data.Dataset yielding (images, one_hot_labels) batches.
    """
    import tensorflow as tf

    file_paths = df["filepath"].tolist()
    labels = [class_to_idx[b] for b in df["breed"]]

    def _parse_image(filename, label):
        img_raw = tf.io.read_file(filename)
        img = tf.io.decode_jpeg(img_raw, channels=3)
        img = tf.image.resize(img, image_size, method="bilinear")
        img = tf.cast(img, tf.float32)
        one_hot = tf.one_hot(label, depth=num_classes)
        return img, one_hot

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), shuffle_buffer))

    dataset = dataset.map(_parse_image, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.batch(batch_size, drop_remainder=False)

    if is_training and augment:
        augmentation_model = create_augmentation_pipeline(image_size)

        def _augment_batch(images, labels_batch):
            return augmentation_model(images, training=True), labels_batch

        dataset = dataset.map(_augment_batch, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def get_dataset_summary(df: pd.DataFrame) -> Dict:
    """Computes summary statistics for the dataset.

    Args:
        df: Dataset DataFrame.

    Returns:
        Dictionary of dataset metrics and counts.
    """
    class_counts = df["breed"].value_counts()
    return {
        "total_images": len(df),
        "total_classes": df["breed"].nunique(),
        "min_images_per_class": int(class_counts.min()),
        "max_images_per_class": int(class_counts.max()),
        "mean_images_per_class": float(class_counts.mean()),
        "median_images_per_class": float(class_counts.median()),
        "top_5_breeds": class_counts.head(5).to_dict(),
        "bottom_5_breeds": class_counts.tail(5).to_dict(),
    }
