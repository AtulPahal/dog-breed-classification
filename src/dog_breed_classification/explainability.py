"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability engine.

Generates visual attention heatmaps indicating the anatomical regions of the dog image
that most influenced the model's breed classification decision.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from PIL import Image


def find_last_conv_layer(model: tf.keras.Model) -> Tuple[Optional[tf.keras.layers.Layer], Optional[str]]:
    """Automatically searches the model graph (including nested backbones)
    to find the deepest 4D convolutional feature layer.

    Args:
        model: tf.keras.Model classifier.

    Returns:
        Tuple of (layer_object, layer_name) or (None, None).
    """
    # 1. Search top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # Nested model (backbone)
            for sub_layer in reversed(layer.layers):
                if any(
                    isinstance(sub_layer, t)
                    for t in [
                        tf.keras.layers.Conv2D,
                        tf.keras.layers.DepthwiseConv2D,
                    ]
                ) or "conv" in sub_layer.name.lower() or "top_activation" in sub_layer.name.lower():
                    return sub_layer, f"{layer.name}/{sub_layer.name}"
        elif isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer, layer.name

    # Fallback search by output shape if names differ
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                try:
                    out_shape = sub_layer.output_shape
                    if isinstance(out_shape, tuple) and len(out_shape) == 4:
                        return sub_layer, f"{layer.name}/{sub_layer.name}"
                except Exception:
                    continue

    return None, None


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: Optional[str] = None,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Computes Grad-CAM heatmap for a single image tensor.

    Args:
        img_array: Input image array of shape (1, height, width, 3).
        model: Full classification model.
        last_conv_layer_name: Optional specific name of conv layer.
        pred_index: Class index to compute gradients for (defaults to top predicted class).

    Returns:
        2D numpy array heatmap with values in [0.0, 1.0].
    """
    # Identify backbone and classification head layers
    backbone = None
    head_layers = []
    found_backbone = False

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            found_backbone = True
        elif found_backbone:
            head_layers.append(layer)

    if backbone is None:
        # Standalone flat model
        return _gradcam_flat_model(img_array, model, pred_index)

    # Locate last conv layer inside backbone
    target_sub_layer = None
    for sub_layer in reversed(backbone.layers):
        if any(
            isinstance(sub_layer, t)
            for t in [tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D]
        ) or any(k in sub_layer.name.lower() for k in ["top_activation", "top_conv", "post_swish", "conv"]):
            try:
                if len(sub_layer.output.shape) == 4:
                    target_sub_layer = sub_layer
                    break
            except Exception:
                continue

    if target_sub_layer is None:
        target_sub_layer = backbone.layers[-1]

    # Build Grad-CAM computation model
    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[target_sub_layer.output, backbone.output],
    )

    # Preprocessing layer if present
    prep_layer = None
    for layer in model.layers:
        if layer == backbone:
            break
        if not isinstance(layer, tf.keras.layers.InputLayer):
            prep_layer = layer

    inputs = tf.cast(img_array, tf.float32)
    if prep_layer is not None:
        inputs = prep_layer(inputs)

    with tf.GradientTape() as tape:
        conv_outputs, backbone_features = grad_model(inputs)
        tape.watch(conv_outputs)

        # Pass through remaining head layers
        x = backbone_features
        for head_layer in head_layers:
            x = head_layer(x)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of top class w.r.t. conv feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        # Fallback uniform heatmap if gradients are zero
        return np.ones(img_array.shape[1:3], dtype=np.float32) * 0.5

    # Global Average Pooling of gradients to calculate feature weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU (only positive influence contributes)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def _gradcam_flat_model(
    img_array: np.ndarray,
    model: tf.keras.Model,
    pred_index: Optional[int] = None,
) -> np.ndarray:
    """Fallback Grad-CAM for models without modular nested backbones."""
    target_layer, _ = find_last_conv_layer(model)
    if target_layer is None:
        return np.ones(img_array.shape[1:3], dtype=np.float32) * 0.5

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        tape.watch(conv_outputs)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        return np.ones(img_array.shape[1:3], dtype=np.float32) * 0.5

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(
    original_image: Union[Image.Image, np.ndarray],
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet",
) -> Tuple[Image.Image, Image.Image]:
    """Blends a Grad-CAM heatmap with the original dog image.

    Args:
        original_image: PIL Image or RGB numpy array [0..255].
        heatmap: 2D float array [0..1] produced by make_gradcam_heatmap.
        alpha: Heatmap blend opacity (0.0=original image, 1.0=pure heatmap).
        colormap_name: Matplotlib colormap ('jet', 'inferno', 'viridis', etc.).

    Returns:
        Tuple of (blended_pil_image, colored_heatmap_pil_image).
    """
    if isinstance(original_image, np.ndarray):
        pil_img = Image.fromarray(original_image.astype(np.uint8))
    else:
        pil_img = original_image.convert("RGB")

    width, height = pil_img.size

    # Resize heatmap to match original image resolution
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    heatmap_pil = Image.fromarray(heatmap_uint8).resize(
        (width, height), Image.Resampling.BILINEAR
    )
    resized_heatmap = np.array(heatmap_pil) / 255.0

    # Colorize using colormap (compatible with matplotlib 3.8+)
    try:
        import matplotlib.pyplot as plt
        colormap = plt.colormaps[colormap_name]
    except Exception:
        import matplotlib.cm as cm
        colormap = cm.get_cmap(colormap_name)

    colored_heatmap = colormap(resized_heatmap)[:, :, :3]  # drop alpha
    colored_heatmap = np.uint8(colored_heatmap * 255)
    # Superimpose heatmap onto original image
    orig_np = np.array(pil_img, dtype=np.float32)
    superimposed = colored_heatmap.astype(np.float32) * alpha + orig_np * (1.0 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)

    blended_img = Image.fromarray(superimposed)
    heatmap_img = Image.fromarray(colored_heatmap)

    return blended_img, heatmap_img
