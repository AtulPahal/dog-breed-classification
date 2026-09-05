"""ONNX Runtime inference engine for Dog Breed Classification.

Runs high-throughput, framework-agnostic inference on ONNX models using ONNX Runtime
with native Apple Silicon CoreML hardware acceleration (CoreMLExecutionProvider) and CPU fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

from dog_breed_classification.config import (
    DEFAULT_IMAGE_SIZE,
    MODELS_DIR,
)
from dog_breed_classification.dataset import load_class_mappings
from dog_breed_classification.export_onnx import export_to_onnx


class ONNXDogBreedPredictor:
    """Predictor using ONNX Runtime for optimized cross-platform inference."""

    def __init__(
        self,
        onnx_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        model_name: str = "efficientnetv2_s",
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        providers: Optional[List[str]] = None,
    ):
        """Initializes ONNX Runtime session.

        Args:
            onnx_path: Path to .onnx model file. If None, auto-resolves or exports.
            class_names_path: Path to class_names.json mapping.
            model_name: Architecture name if auto-exporting.
            image_size: Input image resolution.
            providers: List of ONNX execution providers (defaults to CoreML + CPU).
        """
        self.image_size = image_size
        self.model_name = model_name

        # 1. Resolve Class Mappings
        if class_names_path is None:
            default_json = MODELS_DIR / "class_names.json"
            if default_json.exists():
                class_names_path = default_json
            else:
                from dog_breed_classification.dataset import (
                    get_class_mappings,
                    load_dataset_index,
                )

                df = load_dataset_index()
                _, _, self.class_names = get_class_mappings(df, save_path=default_json)
                class_names_path = default_json

        self.class_to_idx, self.idx_to_class, self.class_names = (
            load_class_mappings(class_names_path)
        )
        self.num_classes = len(self.class_names)

        # 2. Resolve ONNX Model File
        if onnx_path is None:
            candidates = list(MODELS_DIR.glob(f"*{model_name}*.onnx")) or list(
                MODELS_DIR.glob("*.onnx")
            )
            if candidates:
                onnx_path = candidates[0]
            else:
                # Auto-export onnx model
                onnx_path = export_to_onnx(
                    model_name=model_name,
                    image_size=image_size,
                    num_classes=self.num_classes,
                    verbose=False,
                )

        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")

        # 3. Setup Providers (CoreML on Apple Silicon if available)
        if providers is None:
            available = ort.get_available_providers()
            preferred = []
            if "CoreMLExecutionProvider" in available:
                preferred.append("CoreMLExecutionProvider")
            if "CPUExecutionProvider" in available:
                preferred.append("CPUExecutionProvider")
            providers = preferred or available

        self.providers = providers
        self.session = ort.InferenceSession(str(self.onnx_path), providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess_image(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[Image.Image, np.ndarray]:
        """Loads and formats image to [1, height, width, 3] float32 array."""
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        resized_img = pil_img.resize(self.image_size, Image.Resampling.BILINEAR)
        img_arr = np.array(resized_img, dtype=np.float32)
        batch_arr = np.expand_dims(img_arr, axis=0)
        return pil_img, batch_arr

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Classifies a dog image using ONNX Runtime.

        Args:
            image_input: Image path, PIL Image, or numpy array.
            top_k: Number of highest ranking predictions to return.

        Returns:
            Dictionary with 'top_breed', 'top_confidence', and ranked 'predictions'.
        """
        pil_img, batch_arr = self.preprocess_image(image_input)

        # ONNX Runtime forward pass
        raw_outputs = self.session.run([self.output_name], {self.input_name: batch_arr})[0]
        probs = raw_outputs[0]

        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            breed_name = self.class_names[idx]
            prob = float(probs[idx])
            predictions.append(
                {
                    "breed": breed_name,
                    "probability": prob,
                    "percentage": f"{prob * 100:.2f}%",
                    "class_index": int(idx),
                }
            )

        top_pred = predictions[0]

        return {
            "top_breed": top_pred["breed"],
            "top_confidence": top_pred["probability"],
            "top_percentage": top_pred["percentage"],
            "predictions": predictions,
            "original_image": pil_img,
            "runtime": "ONNXRuntime",
            "providers": self.session.get_providers(),
        }

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Performs batch prediction over multiple images with ONNX Runtime."""
        return [self.predict(img, top_k=top_k) for img in images]
