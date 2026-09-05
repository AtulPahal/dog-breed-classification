"""Interactive Gradio Web Application for Dog Breed Classification and Visual Explainability.

Features:
- Live image upload or webcam input.
- Real-time Top-K breed classification with confidence breakdown.
- Dynamic Grad-CAM visual attention heatmaps with customizable blend opacity and colormaps.
- Pre-loaded example gallery from the Stanford Dogs Dataset.
- Breed characteristics and behavioral traits database.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from dog_breed_classification.config import (
    IMAGES_DIR,
    MODELS_DIR,
    SUPPORTED_BACKBONES,
)
from dog_breed_classification.predict import DogBreedPredictor

# Breed characteristics dictionary for educational UI context
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

    # Pick diverse random samples
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
    """Core inference handler for Gradio UI."""
    if image is None:
        return {}, None, "<div style='color: gray;'>Please upload a dog photo.</div>", ""

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
    <div style='background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                padding: 16px; border-radius: 12px; color: white; margin-bottom: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);'>
        <div style='font-size: 14px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.85;'>Top Prediction</div>
        <div style='font-size: 26px; font-weight: bold; margin-top: 4px;'>{top_breed}</div>
        <div style='font-size: 16px; margin-top: 2px; color: #4ade80;'>Confidence: {top_conf}</div>
    </div>
    """

    # 4. Breed Info & Facts
    info = BREED_INFO_DATABASE.get(
        top_breed,
        {
            "group": "Recognized Breed",
            "origin": "International",
            "temperament": "Loyal and intelligent companion",
            "life_span": "10-14 years",
        },
    )
    info_md = f"""
### 🐾 Breed Facts: **{top_breed}**
- **AKC Group:** {info['group']}
- **Origin:** {info['origin']}
- **Temperament:** {info['temperament']}
- **Average Life Span:** {info['life_span']}
    """

    return label_dict, gradcam_img, badge_html, info_md


def build_gradio_app() -> gr.Blocks:
    """Constructs the Gradio application UI."""
    sample_examples = get_sample_example_images(6)

    with gr.Blocks(title="Dog Breed Classifier & Grad-CAM") as demo:
        gr.HTML(
            """
        <div class='hero-header'>
            <h1 style='font-size: 2.2rem; font-weight: 800; margin-bottom: 6px;'>
                🐕 Dog Breed Classification & Visual Explainability
            </h1>
            <p style='font-size: 1.05rem; color: #64748b; margin-top: 0;'>
                Fine-Grained Classification across <b>120 Dog Breeds</b> (Stanford Dogs Dataset)
                with <b>Grad-CAM Attention Heatmaps</b> & <b>Apple Metal GPU Acceleration</b>.
            </p>
        </div>
        """
        )

        with gr.Row():
            # Left Column: Inputs & Controls
            with gr.Column(scale=5):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Dog Photo",
                    sources=["upload", "clipboard", "webcam"],
                )

                with gr.Accordion("⚙️ Model & Explainability Settings", open=True):
                    runtime_selector = gr.Radio(
                        choices=["TensorFlow (Metal GPU)", "ONNX Runtime (CoreML Accelerated)"],
                        value="TensorFlow (Metal GPU)",
                        label="Execution Engine",
                    )
                    model_selector = gr.Dropdown(
                        choices=SUPPORTED_BACKBONES,
                        value="efficientnetv2_s",
                        label="Model Architecture",
                        info="Pre-trained backbones fine-tuned on Stanford Dogs",
                    )
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Top-K Predictions",
                    )
                    enable_gradcam = gr.Checkbox(
                        value=True,
                        label="Enable Grad-CAM Explainability",
                        info="Generates visual heatmap showing image regions the model focused on (TensorFlow mode)",
                    )
                    gradcam_alpha = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.45,
                        step=0.05,
                        label="Heatmap Blend Opacity (Alpha)",
                    )
                    colormap_selector = gr.Dropdown(
                        choices=["jet", "inferno", "magma", "viridis", "plasma"],
                        value="jet",
                        label="Heatmap Colormap",
                    )
                classify_btn = gr.Button("🔍 Identify Dog Breed", variant="primary", size="lg")

                if sample_examples:
                    gr.Examples(
                        examples=sample_examples,
                        inputs=image_input,
                        label="Example Dogs from Dataset",
                    )

            # Right Column: Outputs & Insights
            with gr.Column(scale=6):
                badge_output = gr.HTML()
                label_output = gr.Label(num_top_classes=5, label="Prediction Probabilities")

                with gr.Row():
                    gradcam_output = gr.Image(
                        type="pil",
                        label="Grad-CAM Visual Attention",
                        interactive=False,
                    )

                info_output = gr.Markdown()

        # Connect event triggers
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
    custom_css = """
    .gradio-container { max-width: 1200px !important; margin: auto; }
    .hero-header { text-align: center; margin-bottom: 20px; }
    """
    app = build_gradio_app()
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        inbrowser=inbrowser,
        css=custom_css,
    )


if __name__ == "__main__":
    launch_app()
