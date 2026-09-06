"""Command-Line Interface (CLI) for Dog Breed Classification.

Commands:
- `dog-breed info`: Display dataset statistics, system info, and Apple Silicon Metal GPU status.
- `dog-breed train`: Train a dog classifier using transfer learning and fine-tuning.
- `dog-breed evaluate`: Evaluate model on test set with Top-1/Top-5 accuracy & confusion metrics.
- `dog-breed predict`: Classify a single image with ranked predictions and Grad-CAM heatmap.
- `dog-breed app`: Launch the interactive Gradio web application.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import tensorflow as tf
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dog_breed_classification.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_IMAGE_SIZE,
    MODELS_DIR,
    NUM_CLASSES,
    SUPPORTED_BACKBONES,
    TrainingConfig,
    setup_device,
)
from dog_breed_classification.dataset import (
    get_class_mappings,
    get_dataset_summary,
    load_class_mappings,
    load_dataset_index,
)

console = Console()


def command_info(args: argparse.Namespace) -> None:
    """Displays hardware, dataset, and environment diagnostics."""
    console.print(
        Panel.fit(
            "[bold cyan]Stanford Dogs Breed Classification[/bold cyan]\n"
            "[dim]Fine-Grained Classification with Transfer Learning & Apple Silicon Metal GPU[/dim]",
            border_style="cyan",
        )
    )

    # Hardware & System Table
    sys_table = Table(title="System & Hardware Environment", border_style="blue")
    sys_table.add_column("Value", style="green")

    sys_table.add_row("Python Version", sys.version.split()[0])
    sys_table.add_row("TensorFlow Version", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        sys_table.add_row("Compute Accelerator", f"Apple Silicon Metal GPU ({len(gpus)} device)")
        for idx, gpu in enumerate(gpus):
            sys_table.add_row(f"GPU [{idx}]", str(gpu.name))
    else:
        sys_table.add_row("Compute Accelerator", "CPU (No GPU detected)")

    console.print(sys_table)

    # Dataset Diagnostics Table
    try:
        df = load_dataset_index()
        summary = get_dataset_summary(df)

        data_table = Table(title="Stanford Dogs Dataset Summary", border_style="magenta")
        data_table.add_column("Metric", style="bold yellow")
        data_table.add_column("Count / Value", style="cyan")

        data_table.add_row("Total Images", f"{summary['total_images']:,}")
        data_table.add_row("Target Dog Breeds", f"{summary['total_classes']}")
        data_table.add_row("Min Images per Breed", str(summary["min_images_per_class"]))
        data_table.add_row("Max Images per Breed", str(summary["max_images_per_class"]))
        data_table.add_row("Mean Images per Breed", f"{summary['mean_images_per_class']:.1f}")

        console.print(data_table)

    except Exception as e:
        console.print(f"[bold red]Warning accessing dataset:[/] {e}")


def command_train(args: argparse.Namespace) -> None:
    """Executes the training workflow."""
    from dog_breed_classification.train import train_model

    config = TrainingConfig(
        model_name=args.model_name,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        initial_epochs=args.initial_epochs,
        initial_lr=args.lr,
        fine_tune=not args.no_fine_tune,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        dropout_rate=args.dropout,
        seed=args.seed,
    )

    console.print(f"[bold green]Starting training for architecture:[/] [cyan]{config.model_name}[/]")
    console.print(f"Phase 1: {config.initial_epochs} epochs @ lr={config.initial_lr}")
    if config.fine_tune:
        console.print(f"Phase 2: {config.fine_tune_epochs} epochs @ lr={config.fine_tune_lr}")

    train_model(config=config, verbose=1)
    console.print("[bold green]Training completed successfully![/]")


def command_evaluate(args: argparse.Namespace) -> None:
    """Evaluates a model against the test split."""
    from dog_breed_classification.dataset import create_stratified_splits
    from dog_breed_classification.evaluate import evaluate_model
    from dog_breed_classification.predict import DogBreedPredictor

    console.print("[bold cyan]Loading dataset and model for evaluation...[/]")
    df = load_dataset_index()
    class_to_idx, idx_to_class, class_names = get_class_mappings(df)
    _, _, test_df = create_stratified_splits(df)

    predictor = DogBreedPredictor(
        model_path=args.model_path,
        model_name=args.model_name,
    )

    console.print(f"[bold green]Evaluating on {len(test_df)} test images...[/]")
    metrics = evaluate_model(
        model=predictor.model,
        test_df=test_df,
        class_to_idx=class_to_idx,
        class_names=class_names,
        batch_size=args.batch_size,
    )

    res_table = Table(title="Model Evaluation Summary", border_style="green")
    res_table.add_column("Metric", style="bold yellow")
    res_table.add_column("Score", style="cyan")

    res_table.add_row("Top-1 Accuracy", f"{metrics['top_1_accuracy'] * 100:.2f}%")
    if metrics["top_5_accuracy"] is not None:
        res_table.add_row("Top-5 Accuracy", f"{metrics['top_5_accuracy'] * 100:.2f}%")
    res_table.add_row("Macro F1-Score", f"{metrics['macro_f1']:.4f}")
    res_table.add_row("Weighted F1-Score", f"{metrics['weighted_f1']:.4f}")

    console.print(res_table)


def command_export_onnx(args: argparse.Namespace) -> None:
    """Exports a trained model to ONNX format."""
    from dog_breed_classification.export_onnx import export_to_onnx

    console.print(f"[bold cyan]Exporting model '{args.model_name}' to ONNX...[/]")
    out_path = export_to_onnx(
        model_path=args.model_path,
        model_name=args.model_name,
        output_path=args.output,
        opset=args.opset,
        verbose=True,
    )
    console.print(f"[bold green]ONNX model successfully saved to:[/] [cyan]{out_path}[/]")


def command_predict(args: argparse.Namespace) -> None:
    """Classifies an individual dog image."""
    image_path = Path(args.image)
    if not image_path.exists():
        console.print(f"[bold red]Error:[/] Image not found at {image_path}")
        sys.exit(1)

    console.print(f"[bold cyan]Classifying image:[/] {image_path.name}")

    if args.onnx:
        from dog_breed_classification.predict_onnx import ONNXDogBreedPredictor

        console.print("[dim]Using ONNX Runtime with CoreML / CPU acceleration...[/dim]")
        predictor = ONNXDogBreedPredictor(
            onnx_path=args.model_path,
            model_name=args.model_name,
        )
        result = predictor.predict(image_input=image_path, top_k=args.top_k)
    else:
        from dog_breed_classification.predict import DogBreedPredictor

        predictor = DogBreedPredictor(
            model_path=args.model_path,
            model_name=args.model_name,
        )
        result = predictor.predict(
            image_input=image_path,
            top_k=args.top_k,
            return_gradcam=not args.no_gradcam,
        )

    console.print(
        Panel.fit(
            f"[bold green]Top Prediction:[/] [bold yellow]{result['top_breed']}[/bold yellow]\n"
            f"[bold cyan]Confidence:[/] [bold white]{result['top_percentage']}[/bold white]",
            title="Classification Result",
            border_style="green",
        )
    )

    table = Table(title=f"Top {len(result['predictions'])} Breed Predictions", border_style="cyan")
    table.add_column("Rank", style="dim")
    table.add_column("Breed", style="bold")
    table.add_column("Probability", style="green")

    for rank, item in enumerate(result["predictions"], start=1):
        table.add_row(str(rank), item["breed"], item["percentage"])

    console.print(table)

    if "gradcam_overlay" in result and args.save_cam:
        out_cam_path = Path(args.save_cam)
        out_cam_path.parent.mkdir(parents=True, exist_ok=True)
        result["gradcam_overlay"].save(str(out_cam_path))
        console.print(f"[bold green]Saved Grad-CAM heatmap overlay to:[/] {out_cam_path}")

def command_app(args: argparse.Namespace) -> None:
    """Launches the Gradio Web Application."""
    from dog_breed_classification.app import launch_app

    console.print(f"[bold cyan]Launching Gradio Web App on port {args.port}...[/]")
    launch_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="dog-breed",
        description="Dog Breed Classification & Visual Explainability System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: info
    parser_info = subparsers.add_parser("info", help="Dataset & hardware diagnostics")
    parser_info.set_defaults(func=command_info)

    # Command: train
    parser_train = subparsers.add_parser("train", help="Train a dog breed classifier")
    parser_train.add_argument("--model-name", choices=SUPPORTED_BACKBONES, default="efficientnetv2_s")
    parser_train.add_argument("--initial-epochs", type=int, default=10)
    parser_train.add_argument("--fine-tune-epochs", type=int, default=15)
    parser_train.add_argument("--batch-size", type=int, default=32)
    parser_train.add_argument("--lr", type=float, default=1e-3)
    parser_train.add_argument("--fine-tune-lr", type=float, default=1e-4)
    parser_train.add_argument("--dropout", type=float, default=0.3)
    parser_train.add_argument("--image-size", type=int, default=224)
    parser_train.add_argument("--no-fine-tune", action="store_true")
    parser_train.add_argument("--seed", type=int, default=42)
    parser_train.set_defaults(func=command_train)

    # Command: evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate model on test dataset")
    parser_eval.add_argument("--model-path", type=str, default=None)
    parser_eval.add_argument("--model-name", choices=SUPPORTED_BACKBONES, default="efficientnetv2_s")
    parser_eval.add_argument("--batch-size", type=int, default=32)
    parser_eval.set_defaults(func=command_evaluate)

    # Command: predict
    parser_pred = subparsers.add_parser("predict", help="Predict breed for an image")
    parser_pred.add_argument("image", type=str, help="Path to image file")
    parser_pred.add_argument("--model-path", type=str, default=None)
    parser_pred.add_argument("--model-name", choices=SUPPORTED_BACKBONES, default="efficientnetv2_s")
    parser_pred.add_argument("--top-k", type=int, default=5)
    parser_pred.add_argument("--no-gradcam", action="store_true")
    parser_pred.add_argument("--save-cam", type=str, default=None)
    parser_pred.add_argument("--onnx", action="store_true", help="Use ONNX Runtime engine (CoreML accelerated)")
    parser_pred.set_defaults(func=command_predict)

    # Command: export-onnx
    parser_onnx = subparsers.add_parser("export-onnx", help="Export model to ONNX format")
    parser_onnx.add_argument("--model-path", type=str, default=None)
    parser_onnx.add_argument("--model-name", choices=SUPPORTED_BACKBONES, default="efficientnetv2_s")
    parser_onnx.add_argument("--output", type=str, default=None)
    parser_onnx.add_argument("--opset", type=int, default=13)
    parser_onnx.set_defaults(func=command_export_onnx)
    # Command: app
    parser_app = subparsers.add_parser("app", help="Launch interactive Gradio Web UI")
    parser_app.add_argument("--host", type=str, default="127.0.0.1")
    parser_app.add_argument("--port", type=int, default=7860)
    parser_app.add_argument("--share", action="store_true")
    parser_app.add_argument("--no-browser", action="store_true")
    parser_app.set_defaults(func=command_app)

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
