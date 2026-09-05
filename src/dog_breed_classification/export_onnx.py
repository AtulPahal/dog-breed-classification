"""ONNX model exporter for Dog Breed Classification.

Exports trained Keras models to standard ONNX (Open Neural Network Exchange) format
for high-performance cross-platform deployment on macOS (via CoreML Execution Provider),
Linux, Windows, iOS, Android, and web runtimes (ONNX Runtime Web / WebAssembly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import onnx
import tensorflow as tf
import tf2onnx

from dog_breed_classification.config import (
    DEFAULT_IMAGE_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
)
from dog_breed_classification.models import build_dog_classifier


def export_to_onnx(
    model: Optional[tf.keras.Model] = None,
    model_path: Optional[Union[str, Path]] = None,
    model_name: str = "efficientnetv2_s",
    output_path: Optional[Union[str, Path]] = None,
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    opset: int = 13,
    verbose: bool = True,
) -> Path:
    """Converts a Keras Dog Breed Classification model to ONNX format.

    Args:
        model: Loaded tf.keras.Model instance (if None, loads from model_path).
        model_path: Path to saved .keras or weights file.
        model_name: Architecture identifier if instantiating from scratch.
        output_path: Destination path for .onnx file (defaults to artifacts/models/<name>.onnx).
        image_size: Image input resolution (height, width).
        num_classes: Number of dog breeds (120).
        opset: ONNX operator set version (default: 13 for wide compatibility).
        verbose: Whether to print export details.

    Returns:
        Path to the generated .onnx model file.
    """
    if model is None:
        if model_path and Path(model_path).exists():
            if verbose:
                print(f"[*] Loading model from {model_path}...")
            try:
                model = tf.keras.models.load_model(str(model_path))
            except Exception:
                model = build_dog_classifier(
                    model_name=model_name,
                    input_shape=(image_size[0], image_size[1], 3),
                    num_classes=num_classes,
                )
                model.load_weights(str(model_path))
        else:
            if verbose:
                print(
                    f"[*] Building fresh pre-trained '{model_name}' model for ONNX export..."
                )
            model = build_dog_classifier(
                model_name=model_name,
                input_shape=(image_size[0], image_size[1], 3),
                num_classes=num_classes,
                weights="imagenet",
            )

    if output_path is None:
        output_path = MODELS_DIR / f"dog_classifier_{model_name}.onnx"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[*] Converting model to ONNX (opset={opset})...")

    input_signature = [
        tf.TensorSpec(
            shape=[1, image_size[0], image_size[1], 3],
            dtype=tf.float32,
            name="image_input",
        )
    ]

    onnx_model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
    )

    # Validate ONNX model integrity
    onnx.checker.check_model(onnx_model_proto)

    with open(output_path, "wb") as f:
        f.write(onnx_model_proto.SerializeToString())

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    if verbose:
        print(f"[✅] Successfully exported ONNX model!")
        print(f"    - File: {output_path.resolve()}")
        print(f"    - Size: {file_size_mb:.2f} MB")
        print(f"    - Input: shape [1, {image_size[0]}, {image_size[1]}, 3], float32 (RGB [0..255])")
        print(f"    - Output: shape [1, {num_classes}], float32 (Softmax Probabilities)")

    return output_path
