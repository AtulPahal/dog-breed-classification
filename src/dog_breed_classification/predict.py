"""Inference engine and prediction utilities for Dog Breed Classification.

Provides the DogBreedPredictor class for:
- Single image inference with Top-K ranked breed probabilities.
- Batch inference.
- Automatic Grad-CAM visual attention generation.
- Support for file paths, URLs, PIL Images, and NumPy arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from PIL import Image

from dog_breed_classification.config import (
    DEFAULT_IMAGE_SIZE,
    MODELS_DIR,
    setup_device,
)
from dog_breed_classification.dataset import load_class_mappings
from dog_breed_classification.explainability import (
    make_gradcam_heatmap,
    overlay_gradcam,
)
from dog_breed_classification.models import build_dog_classifier


class DogBreedPredictor:
    """End-to-end predictor for classifying dog breeds with visual explainability."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        class_names_path: Optional[Union[str, Path]] = None,
        model_name: str = "efficientnetv2_s",
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        device_setup: bool = True,
    ):
        """Initializes the dog breed predictor.

        Args:
            model_path: Path to .keras or weights file. If None, looks in models_dir.
            class_names_path: Path to class_names.json mapping file.
            model_name: Default architecture to instantiate if loading weights.
            image_size: Input image resolution (height, width).
            device_setup: Whether to configure Apple Metal GPU device.
        """
        if device_setup:
            setup_device(verbose=False)

        self.image_size = image_size
        self.model_name = model_name

        # 1. Resolve Class Mappings
        if class_names_path is None:
            default_json = MODELS_DIR / "class_names.json"
            if default_json.exists():
                class_names_path = default_json
            else:
                # Generate from dataset if available
                from dog_breed_classification.dataset import (
                    get_class_mappings,
                    load_dataset_index,
                )

                df = load_dataset_index()
                _, _, self.class_names = get_class_mappings(
                    df, save_path=default_json
                )
                class_names_path = default_json

        self.class_to_idx, self.idx_to_class, self.class_names = (
            load_class_mappings(class_names_path)
        )
        self.num_classes = len(self.class_names)

        # 2. Resolve Model
        if model_path is None:
            candidates = list(MODELS_DIR.glob(f"*{model_name}*.keras")) or list(
                MODELS_DIR.glob("*.keras")
            )
            if candidates:
                model_path = candidates[0]

        if model_path and Path(model_path).exists():
            model_path_str = str(model_path)
            try:
                self.model = tf.keras.models.load_model(model_path_str)
            except Exception:
                # If weights file or custom architecture
                self.model = build_dog_classifier(
                    model_name=model_name,
                    input_shape=(image_size[0], image_size[1], 3),
                    num_classes=self.num_classes,
                )
                self.model.load_weights(model_path_str)
        else:
            # Initialize with ImageNet pre-trained weights for instant usage
            self.model = build_dog_classifier(
                model_name=model_name,
                input_shape=(image_size[0], image_size[1], 3),
                num_classes=self.num_classes,
                weights="imagenet",
            )

    def preprocess_image(
        self, image_input: Union[str, Path, Image.Image, np.ndarray]
    ) -> Tuple[Image.Image, np.ndarray]:
        """Loads and normalizes an input image to the required shape.

        Args:
            image_input: Image filepath, PIL Image, or numpy array.

        Returns:
            Tuple of (original_pil_image, batch_tensor_array).
        """
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
        return_gradcam: bool = True,
        gradcam_alpha: float = 0.45,
    ) -> Dict[str, Any]:
        """Predicts the dog breed for a single image with confidence scores and Grad-CAM.

        Args:
            image_input: Filepath, PIL Image, or numpy array.
            top_k: Number of highest ranking predictions to return.
            return_gradcam: Whether to compute visual attention heatmap.
            gradcam_alpha: Heatmap overlay blending factor.

        Returns:
            Dictionary containing:
            - 'top_breed': string name of most probable breed.
            - 'top_confidence': float probability [0.0..1.0].
            - 'predictions': list of top_k dicts with breed and probability.
            - 'gradcam_overlay': PIL Image with heatmap (if return_gradcam).
            - 'gradcam_heatmap': PIL Image of standalone heatmap (if return_gradcam).
            - 'original_image': Original PIL Image.
        """
        pil_img, batch_arr = self.preprocess_image(image_input)

        # Forward pass
        probs = self.model.predict(batch_arr, verbose=0)[0]

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

        result = {
            "top_breed": top_pred["breed"],
            "top_confidence": top_pred["probability"],
            "top_percentage": top_pred["percentage"],
            "predictions": predictions,
            "original_image": pil_img,
        }

        if return_gradcam:
            try:
                heatmap = make_gradcam_heatmap(
                    img_array=batch_arr,
                    model=self.model,
                    pred_index=top_pred["class_index"],
                )
                overlay_img, heatmap_img = overlay_gradcam(
                    original_image=pil_img,
                    heatmap=heatmap,
                    alpha=gradcam_alpha,
                )
                result["gradcam_overlay"] = overlay_img
                result["gradcam_heatmap"] = heatmap_img
            except Exception as e:
                result["gradcam_error"] = str(e)

        return result

    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Performs batch prediction over multiple images.

        Args:
            images: List of image inputs.
            top_k: Top K predictions per image.

        Returns:
            List of prediction dictionaries.
        """
        results = []
        for img in images:
            results.append(self.predict(img, top_k=top_k, return_gradcam=False))
        return results
