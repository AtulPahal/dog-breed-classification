"""Model architectures and transfer learning backends for Dog Breed Classification.

Supports pre-trained backbones (EfficientNetV2-S, EfficientNetV2-B0, MobileNetV3-Large, ResNet50V2)
with integrated preprocessing, custom regularized classification heads, and fine-tuning controls.
"""

from __future__ import annotations

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from dog_breed_classification.config import (
    DEFAULT_IMAGE_SIZE,
    NUM_CLASSES,
    SUPPORTED_BACKBONES,
)


def get_backbone(
    model_name: str,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    weights: str = "imagenet",
) -> Tuple[tf.keras.Model, Optional[layers.Layer]]:
    """Loads pre-trained backbone model and its respective preprocessing layer.

    Args:
        model_name: Name of architecture ('efficientnetv2_s', 'efficientnetv2_b0',
                    'mobilenetv3_large', 'resnet50v2').
        input_shape: Input image shape (height, width, channels).
        weights: Pretrained weights ('imagenet' or None).

    Returns:
        Tuple of (base_model, preprocessing_layer).
    """
    model_name_lower = model_name.lower().replace("-", "_")

    if model_name_lower == "efficientnetv2_s":
        base_model = tf.keras.applications.EfficientNetV2S(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
        preprocess_layer = layers.Lambda(
            tf.keras.applications.efficientnet_v2.preprocess_input,
            name="efficientnetv2_preprocess",
        )

    elif model_name_lower == "efficientnetv2_b0":
        base_model = tf.keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
        preprocess_layer = layers.Lambda(
            tf.keras.applications.efficientnet_v2.preprocess_input,
            name="efficientnetv2_preprocess",
        )

    elif model_name_lower == "mobilenetv3_large":
        base_model = tf.keras.applications.MobileNetV3Large(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
        preprocess_layer = layers.Lambda(
            tf.keras.applications.mobilenet_v3.preprocess_input,
            name="mobilenetv3_preprocess",
        )

    elif model_name_lower == "resnet50v2":
        base_model = tf.keras.applications.ResNet50V2(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
        preprocess_layer = layers.Lambda(
            tf.keras.applications.resnet_v2.preprocess_input,
            name="resnetv2_preprocess",
        )

    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Supported options are: {SUPPORTED_BACKBONES}"
        )

    base_model._name = f"backbone_{model_name_lower}"
    return base_model, preprocess_layer


def build_dog_classifier(
    model_name: str = "efficientnetv2_s",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.3,
    dense_units: int = 512,
    l2_reg: float = 1e-4,
    freeze_backbone: bool = True,
    weights: str = "imagenet",
) -> tf.keras.Model:
    """Builds a complete dog breed classification model with pre-trained backbone
    and regularized classification head.

    Args:
        model_name: Backbone architecture identifier.
        input_shape: Image input dimensions.
        num_classes: Number of target dog breeds (default: 120).
        dropout_rate: Dropout probability in dense layer.
        dense_units: Number of neurons in intermediate dense layer.
        l2_reg: L2 weight regularization coefficient.
        freeze_backbone: Whether to freeze backbone weights initially.
        weights: Pre-trained weight set.

    Returns:
        Compiled or uncompiled tf.keras.Model.
    """
    base_model, preprocess_fn = get_backbone(
        model_name=model_name,
        input_shape=input_shape,
        weights=weights,
    )

    if freeze_backbone:
        base_model.trainable = False

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = preprocess_fn(inputs) if preprocess_fn else inputs
    x = base_model(x, training=False)  # keep batchnorm in inference mode
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)

    if dense_units > 0:
        x = layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg),
            name="head_dense",
        )(x)
        x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="predictions",
    )(x)

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"DogClassifier_{model_name}",
    )
    return model


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    label_smoothing: float = 0.1,
    weight_decay: float = 1e-4,
) -> tf.keras.Model:
    """Compiles the model with Adam optimizer, smoothed Categorical Crossentropy,
    and Top-1 / Top-5 accuracy metrics.

    Args:
        model: tf.keras.Model instance.
        learning_rate: Initial learning rate.
        label_smoothing: Label smoothing epsilon.
        weight_decay: L2 weight decay for optimizer.

    Returns:
        Compiled model.
    """
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
    except Exception:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    ]

    # Add Top-5 accuracy only if model has more than 5 classes
    try:
        out_dim = model.output_shape[-1]
        if out_dim and out_dim >= 5:
            metrics.append(
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_accuracy")
            )
    except Exception:
        pass

    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=label_smoothing,
        name="categorical_crossentropy",
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
    )
    return model

def set_backbone_trainable(
    model: tf.keras.Model,
    unfreeze_layers: int = 50,
    learning_rate: float = 1e-4,
    label_smoothing: float = 0.1,
) -> tf.keras.Model:
    """Unfreezes the top N layers of the backbone model for fine-tuning while
    maintaining BatchNormalization layers in inference mode to prevent statistic drift.

    Args:
        model: Dog breed classifier model.
        unfreeze_layers: Number of top backbone layers to unfreeze (pass -1 for all).
        learning_rate: Reduced learning rate for fine-tuning.
        label_smoothing: Label smoothing parameter.

    Returns:
        Re-compiled model ready for fine-tuning.
    """
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
        elif hasattr(layer, "layers") and len(layer.layers) > 1:
            backbone = layer
            break

    if backbone is None:
        raise ValueError(
            f"No backbone model layer found in model layers: {[l.name for l in model.layers]}"
        )
    backbone.trainable = True

    if unfreeze_layers > 0 and unfreeze_layers < len(backbone.layers):
        freeze_until = len(backbone.layers) - unfreeze_layers
        for layer in backbone.layers[:freeze_until]:
            layer.trainable = False
        for layer in backbone.layers[freeze_until:]:
            # Keep BatchNorm frozen even when unfreezing other layers for fine-tuning stability
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
    elif unfreeze_layers == -1:
        # All layers trainable except BatchNorm
        for layer in backbone.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    return compile_model(
        model,
        learning_rate=learning_rate,
        label_smoothing=label_smoothing,
    )
