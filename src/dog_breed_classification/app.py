"""Interactive Gradio Web Application for Dog Breed Classification and Visual Explainability.

Implements a responsive, mobile-first Bento Grid interface with semantic layout,
strict typography standards, and zero emojis.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from dog_breed_classification.config import (
    IMAGES_DIR,
    MODELS_DIR,
    SUPPORTED_BACKBONES,
)
from dog_breed_classification.predict import DogBreedPredictor

# Breed characteristics database for educational UI context
BREED_INFO_DATABASE = {
    "Chihuahua": {
        "group": "Toy",
        "origin": "Mexico",
        "temperament": "Charming, Graceful, Sassy",
        "life_span": "14-16 years",
    },
    "Golden Retriever": {
        "group": "Sporting",
        "origin": "Scotland",
        "temperament": "Intelligent, Friendly, Devoted",
        "life_span": "10-12 years",
    },
    "Labrador Retriever": {
        "group": "Sporting",
        "origin": "Canada (Newfoundland)",
        "temperament": "Friendly, Active, Outgoing",
        "life_span": "11-13 years",
    },
    "German Shepherd": {
        "group": "Herding",
        "origin": "Germany",
        "temperament": "Confident, Courageous, Smart",
        "life_span": "7-10 years",
    },
    "Siberian Husky": {
        "group": "Working",
        "origin": "Siberia, Russia",
        "temperament": "Loyal, Mischievous, Outgoing",
        "life_span": "12-14 years",
    },
    "Bernese Mountain Dog": {
        "group": "Working",
        "origin": "Switzerland",
        "temperament": "Good-natured, Calm, Strong",
        "life_span": "7-10 years",
    },
    "Pug": {
        "group": "Toy",
        "origin": "China",
        "temperament": "Loving, Charming, Mischievous",
        "life_span": "13-15 years",
    },
    "French Bulldog": {
        "group": "Non-Sporting",
        "origin": "France",
        "temperament": "Playful, Adaptable, Smart",
        "life_span": "10-12 years",
    },
    "Rottweiler": {
        "group": "Working",
        "origin": "Germany",
        "temperament": "Loyal, Loving, Confident Guardian",
        "life_span": "9-10 years",
    },
    "Border Collie": {
        "group": "Herding",
        "origin": "United Kingdom",
        "temperament": "Tenacious, Keen, Energetic, Highly Intelligent",
        "life_span": "12-15 years",
    },
    "Beagle": {
        "group": "Hound",
        "origin": "United Kingdom",
        "temperament": "Friendly, Curious, Merry",
        "life_span": "10-15 years",
    },
    "Doberman": {
        "group": "Working",
        "origin": "Germany",
        "temperament": "Loyal, Fearless, Alert",
        "life_span": "10-12 years",
    },
    "Pomeranian": {
        "group": "Toy",
        "origin": "Germany / Poland",
        "temperament": "Inquisitive, Lively, Bold",
        "life_span": "12-16 years",
    },
    "Great Dane": {
        "group": "Working",
        "origin": "Germany",
        "temperament": "Friendly, Patient, Dependable",
        "life_span": "7-10 years",
    },
    "Cardigan": {
        "group": "Herding",
        "origin": "Wales",
        "temperament": "Affectionate, Loyal, Alert",
        "life_span": "12-15 years",
    },
}

# Predictor instance caches
_TF_PREDICTOR_CACHE: Dict[str, DogBreedPredictor] = {}
_ONNX_PREDICTOR_CACHE: Dict[str, Any] = {}


def get_cached_predictor(model_name: str, runtime: str = "TensorFlow (Metal GPU)"):
    """Retrieves or instantiates a cached predictor for the selected backbone and runtime."""
    if runtime.startswith("ONNX"):
        from dog_breed_classification.predict_onnx import ONNXDogBreedPredictor

        if model_name not in _ONNX_PREDICTOR_CACHE:
            _ONNX_PREDICTOR_CACHE[model_name] = ONNXDogBreedPredictor(model_name=model_name)
        return _ONNX_PREDICTOR_CACHE[model_name]
    else:
        if model_name not in _TF_PREDICTOR_CACHE:
            _TF_PREDICTOR_CACHE[model_name] = DogBreedPredictor(model_name=model_name)
        return _TF_PREDICTOR_CACHE[model_name]


def get_sample_example_images(num_samples: int = 6) -> List[str]:
    """Selects representative sample dog images from the dataset directory."""
    if not IMAGES_DIR.exists():
        return []

    image_paths = list(IMAGES_DIR.glob("*/*.jpg"))
    if not image_paths:
        return []

    random.seed(42)
    selected = random.sample(image_paths, min(num_samples, len(image_paths)))
    return [str(p.resolve()) for p in selected]


def classify_dog_image(
    image: Optional[Image.Image],
    model_name: str,
    runtime: str,
    top_k: int,
    enable_gradcam: bool,
    gradcam_alpha: float,
    colormap_name: str,
) -> Tuple[Dict[str, float], Optional[Image.Image], str, str]:
    """Core inference handler for Gradio UI.

    Args:
        image: User uploaded image.
        model_name: Selected model architecture.
        runtime: Selected execution engine (TensorFlow or ONNX).
        top_k: Number of predictions to return.
        enable_gradcam: Whether to render Grad-CAM heatmap.
        gradcam_alpha: Heatmap overlay blending factor.
        colormap_name: Color scheme for Grad-CAM.

    Returns:
        Tuple of (label_dict, gradcam_image, top_badge_html, breed_info_html).
    """
    if image is None:
        placeholder_badge = """
        <div class="bento-badge-card empty-state">
            <span class="badge-tag">Status</span>
            <div class="badge-title">Awaiting Input</div>
            <div class="badge-sub">Upload or select a dog photo to classify.</div>
        </div>
        """
        placeholder_info = """
        <div class="bento-info-card empty-state">
            <div class="info-row"><span class="info-label">AKC Group</span><span class="info-val">-</span></div>
            <div class="info-row"><span class="info-label">Origin</span><span class="info-val">-</span></div>
            <div class="info-row"><span class="info-label">Temperament</span><span class="info-val">-</span></div>
            <div class="info-row"><span class="info-label">Life Span</span><span class="info-val">-</span></div>
        </div>
        """
        return {}, None, placeholder_badge, placeholder_info

    predictor = get_cached_predictor(model_name, runtime=runtime)
    is_onnx = runtime.startswith("ONNX")

    if is_onnx:
        result = predictor.predict(image_input=image, top_k=top_k)
    else:
        result = predictor.predict(
            image_input=image,
            top_k=top_k,
            return_gradcam=enable_gradcam,
            gradcam_alpha=gradcam_alpha,
        )

    # 1. Format probabilities for gr.Label
    label_dict = {
        item["breed"]: float(item["probability"]) for item in result["predictions"]
    }

    # 2. Grad-CAM visual
    gradcam_img = result.get("gradcam_overlay", None)

    # 3. Top Prediction Highlight Badge
    top_breed = result["top_breed"]
    top_conf = result["top_percentage"]
    badge_html = f"""
    <div class="bento-badge-card active-state">
        <div class="badge-header">
            <span class="badge-tag">Top Classification</span>
            <span class="badge-conf-pill">{top_conf} Confidence</span>
        </div>
        <div class="badge-title">{top_breed}</div>
        <div class="badge-sub">Evaluated across 120 Stanford Dogs classes</div>
    </div>
    """

    # 4. Breed Info & Facts Card
    info = BREED_INFO_DATABASE.get(
        top_breed,
        {
            "group": "Recognized Breed",
            "origin": "International",
            "temperament": "Loyal and intelligent companion",
            "life_span": "10-14 years",
        },
    )
    info_html = f"""
    <div class="bento-info-card active-state">
        <div class="info-title">Breed Profile: {top_breed}</div>
        <div class="info-grid">
            <div class="info-row"><span class="info-label">AKC Group</span><span class="info-val">{info['group']}</span></div>
            <div class="info-row"><span class="info-label">Origin</span><span class="info-val">{info['origin']}</span></div>
            <div class="info-row"><span class="info-label">Temperament</span><span class="info-val">{info['temperament']}</span></div>
            <div class="info-row"><span class="info-label">Life Span</span><span class="info-val">{info['life_span']}</span></div>
        </div>
    </div>
    """

    return label_dict, gradcam_img, badge_html, info_html


BENTO_CUSTOM_CSS = """
/* Reset and Global Container */
:root {
    --bento-bg: #0f172a;
    --bento-card-bg: rgba(30, 41, 59, 0.7);
    --bento-card-border: rgba(255, 255, 255, 0.08);
    --bento-accent: #3b82f6;
    --bento-accent-glow: rgba(59, 130, 246, 0.15);
    --bento-text-main: #f8fafc;
    --bento-text-muted: #94a3b8;
    --font-sans: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", Roboto, Helvetica, sans-serif;
}

body, .gradio-container {
    font-family: var(--font-sans) !important;
    max-width: 1380px !important;
    margin: 0 auto !important;
    padding: 12px 16px !important;
    background-color: var(--bento-bg) !important;
    color: var(--bento-text-main) !important;
}

/* Header & System Status Bar */
.bento-header {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 16px;
    padding: 18px 24px;
    background: var(--bento-card-bg);
    border: 1px solid var(--bento-card-border);
    border-radius: 14px;
    backdrop-filter: blur(12px);
}

@media (min-width: 768px) {
    .bento-header {
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
    }
}

.bento-header-left h1 {
    font-size: 20px !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin: 0 !important;
    color: #ffffff !important;
}

.bento-header-left p {
    font-size: 13px !important;
    color: var(--bento-text-muted) !important;
    margin: 4px 0 0 0 !important;
}

.bento-header-right {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #e2e8f0;
}

.status-pill.highlight {
    background: var(--bento-accent-glow);
    border-color: rgba(59, 130, 246, 0.3);
    color: #60a5fa;
}

/* Bento Cards */
.bento-card {
    background: var(--bento-card-bg) !important;
    border: 1px solid var(--bento-card-border) !important;
    border-radius: 14px !important;
    padding: 16px !important;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.3) !important;
    backdrop-filter: blur(12px) !important;
}

.bento-card-header {
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    font-weight: 700 !important;
    color: var(--bento-text-muted) !important;
    margin-bottom: 12px !important;
}

/* Prediction Badge */
.bento-badge-card {
    padding: 18px 20px;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 14px;
}

.bento-badge-card.active-state {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.5) 0%, rgba(15, 23, 42, 0.8) 100%);
    border-color: rgba(59, 130, 246, 0.3);
}

.badge-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.badge-tag {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
}

.badge-conf-pill {
    background: rgba(34, 197, 94, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.3);
    color: #4ade80;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 700;
}

.badge-title {
    font-size: 22px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.01em;
}

.badge-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 2px;
}

/* Info Card */
.bento-info-card {
    padding: 16px;
    border-radius: 12px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.info-title {
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #cbd5e1;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.info-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.info-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 13px;
}

.info-label {
    color: #64748b;
    font-weight: 600;
}

.info-val {
    color: #f1f5f9;
    font-weight: 600;
    text-align: right;
}

/* Button Styling */
button.primary-action-btn {
    background: #2563eb !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 20px !important;
    color: #ffffff !important;
    transition: all 0.15s ease-in-out !important;
}

button.primary-action-btn:hover {
    background: #1d4ed8 !important;
}
"""


def build_gradio_app() -> gr.Blocks:
    """Constructs the semantic Bento Grid Gradio application."""
    sample_examples = get_sample_example_images(6)

    with gr.Blocks(title="Dog Breed Classifier & Grad-CAM") as demo:
        # 1. Bento Top Header & Telemetry Bar
        gr.HTML(
            """
        <div class="bento-header">
            <div class="bento-header-left">
                <h1>Dog Breed Classification & Visual Explainability</h1>
                <p>Stanford Dogs Dataset (120 Breeds) - Deep Transfer Learning & Explainable AI</p>
            </div>
            <div class="bento-header-right">
                <span class="status-pill highlight">91.55% Top-1 Accuracy</span>
                <span class="status-pill highlight">99.45% Top-5 Accuracy</span>
                <span class="status-pill">Apple Metal GPU Active</span>
                <span class="status-pill">CoreML ONNX Runtime</span>
            </div>
        </div>
        """
        )

        # 2. Main Bento Grid (2-Column Responsive Layout)
        with gr.Row(equal_height=False):
            # Left Bento Column: Input, Examples, and Inference Controls
            with gr.Column(scale=5, elem_classes=["bento-card"]):
                gr.HTML('<div class="bento-card-header">Input Source</div>')

                image_input = gr.Image(
                    type="pil",
                    label="Image Upload or Capture",
                    sources=["upload", "clipboard", "webcam"],
                    elem_classes=["input-image-box"],
                )

                classify_btn = gr.Button(
                    "Execute Classification",
                    variant="primary",
                    elem_classes=["primary-action-btn"],
                )

                if sample_examples:
                    gr.Examples(
                        examples=sample_examples,
                        inputs=image_input,
                        label="Dataset Reference Samples",
                    )

                with gr.Accordion("Inference Engine & Model Parameters", open=False):
                    runtime_selector = gr.Radio(
                        choices=["TensorFlow (Metal GPU)", "ONNX Runtime (CoreML Accelerated)"],
                        value="TensorFlow (Metal GPU)",
                        label="Execution Engine",
                    )
                    model_selector = gr.Dropdown(
                        choices=SUPPORTED_BACKBONES,
                        value="efficientnetv2_s",
                        label="Architecture",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top-K Ranks",
                    )
                    enable_gradcam = gr.Checkbox(
                        value=True,
                        label="Compute Grad-CAM Attention Heatmap",
                    )
                    gradcam_alpha = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.45,
                        step=0.05,
                        label="Grad-CAM Overlay Blend (Alpha)",
                    )
                    colormap_selector = gr.Dropdown(
                        choices=["jet", "inferno", "magma", "viridis", "plasma"],
                        value="jet",
                        label="Heatmap Colormap",
                    )

            # Right Bento Column: Verdict, Probability Distribution, Grad-CAM, & Profile
            with gr.Column(scale=6, elem_classes=["bento-card"]):
                gr.HTML('<div class="bento-card-header">Classification Output</div>')

                badge_output = gr.HTML(
                    """
                <div class="bento-badge-card empty-state">
                    <span class="badge-tag">Status</span>
                    <div class="badge-title">Awaiting Input</div>
                    <div class="badge-sub">Upload or select a dog photo to classify.</div>
                </div>
                """
                )

                label_output = gr.Label(
                    num_top_classes=5,
                    label="Ranked Class Probabilities",
                )

                gr.HTML('<div class="bento-card-header" style="margin-top: 16px;">Grad-CAM Visual Attention</div>')
                gradcam_output = gr.Image(
                    type="pil",
                    label="Spatial Attention Heatmap",
                    interactive=False,
                )

                gr.HTML('<div class="bento-card-header" style="margin-top: 16px;">Breed Characteristics</div>')
                info_output = gr.HTML(
                    """
                <div class="bento-info-card empty-state">
                    <div class="info-row"><span class="info-label">AKC Group</span><span class="info-val">-</span></div>
                    <div class="info-row"><span class="info-label">Origin</span><span class="info-val">-</span></div>
                    <div class="info-row"><span class="info-label">Temperament</span><span class="info-val">-</span></div>
                    <div class="info-row"><span class="info-label">Life Span</span><span class="info-val">-</span></div>
                </div>
                """
                )

        # 3. Bind Event Listeners
        classify_btn.click(
            fn=classify_dog_image,
            inputs=[
                image_input,
                model_selector,
                runtime_selector,
                top_k_slider,
                enable_gradcam,
                gradcam_alpha,
                colormap_selector,
            ],
            outputs=[label_output, gradcam_output, badge_output, info_output],
        )

        image_input.change(
            fn=classify_dog_image,
            inputs=[
                image_input,
                model_selector,
                runtime_selector,
                top_k_slider,
                enable_gradcam,
                gradcam_alpha,
                colormap_selector,
            ],
            outputs=[label_output, gradcam_output, badge_output, info_output],
        )

    return demo


def launch_app(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
):
    """Launches the Gradio web server."""
    app = build_gradio_app()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
        css=BENTO_CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch_app()
