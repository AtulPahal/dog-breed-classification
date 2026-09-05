"""Comprehensive end-to-end test suite for Dog Breed Classification."""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from dog_breed_classification.config import (
    IMAGES_DIR,
    NUM_CLASSES,
    SUPPORTED_BACKBONES,
    TrainingConfig,
    setup_device,
)
from dog_breed_classification.dataset import (
    clean_breed_name,
    load_dataset_index,
    get_class_mappings,
    create_stratified_splits,
    build_tf_dataset,
    create_augmentation_pipeline,
    get_dataset_summary,
)
from dog_breed_classification.models import (
    build_dog_classifier,
    compile_model,
    set_backbone_trainable,
)
from dog_breed_classification.train import train_model
from dog_breed_classification.evaluate import evaluate_model
from dog_breed_classification.explainability import (
    make_gradcam_heatmap,
    overlay_gradcam,
)
from dog_breed_classification.predict import DogBreedPredictor
from dog_breed_classification.app import build_gradio_app


def test_clean_breed_name():
    print("[Test 1] Testing clean_breed_name...")
    assert clean_breed_name("n02085620-Chihuahua") == "Chihuahua"
    assert clean_breed_name("n02099601-golden_retriever") == "Golden Retriever"
    assert clean_breed_name("n02086240-Shih-Tzu") == "Shih-Tzu"
    assert clean_breed_name("n02107683-Bernese_mountain_dog") == "Bernese Mountain Dog"
    print("  -> Passed!")


def test_dataset_indexing():
    print("[Test 2] Testing dataset indexing & splits...")
    df = load_dataset_index()
    assert len(df) == 20580, f"Expected 20580 images, got {len(df)}"
    assert df["breed"].nunique() == 120, f"Expected 120 breeds, got {df['breed'].nunique()}"

    class_to_idx, idx_to_class, class_names = get_class_mappings(df)
    assert len(class_names) == 120
    assert len(class_to_idx) == 120

    train_df, val_df, test_df = create_stratified_splits(df, 0.7, 0.15, 0.15)
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
    assert train_df["breed"].nunique() == 120
    assert val_df["breed"].nunique() == 120
    assert test_df["breed"].nunique() == 120

    # Test tf.data pipeline
    sub_df = train_df.head(8)
    ds = build_tf_dataset(sub_df, class_to_idx, batch_size=4, is_training=True, augment=True)
    for bx, by in ds.take(1):
        assert bx.shape == (4, 224, 224, 3)
        assert by.shape == (4, 120)
    print("  -> Passed!")


def test_model_architectures():
    print("[Test 3] Testing model architectures...")
    for backbone in ["efficientnetv2_b0", "mobilenetv3_large"]:
        m = build_dog_classifier(model_name=backbone, freeze_backbone=True)
        m = compile_model(m, learning_rate=1e-3)
        assert m.output_shape == (None, 120)
        m = set_backbone_trainable(m, unfreeze_layers=20, learning_rate=1e-4)
    print("  -> Passed!")


def test_micro_training_and_evaluation():
    print("[Test 4] Testing micro-training loop & evaluation...")
    temp_dir = Path(tempfile.mkdtemp())
    try:
        df = load_dataset_index()
        # Take 4 breeds with 15 samples each for fast unit test
        selected_breeds = df["breed"].unique()[:4]
        micro_df = (
            df[df["breed"].isin(selected_breeds)]
            .groupby("breed")
            .head(15)
            .reset_index(drop=True)
        )
        cfg = TrainingConfig(
            model_name="efficientnetv2_b0",
            initial_epochs=1,
            fine_tune_epochs=1,
            fine_tune_layers=10,
            batch_size=16,
            models_dir=temp_dir / "models",
            checkpoints_dir=temp_dir / "checkpoints",
            logs_dir=temp_dir / "logs",
            plots_dir=temp_dir / "plots",
            reports_dir=temp_dir / "reports",
        )
        
        model, history, (tr, val, te) = train_model(config=cfg, df=micro_df, verbose=0)
        assert "accuracy" in history
        assert len(history["accuracy"]) == 2  # 1 initial + 1 fine-tune
        
        class_to_idx, _, class_names = get_class_mappings(micro_df)
        metrics = evaluate_model(
            model=model,
            test_df=te,
            class_to_idx=class_to_idx,
            class_names=class_names,
            batch_size=16,
            reports_dir=temp_dir / "reports",
            plots_dir=temp_dir / "plots",
        )
        assert "top_1_accuracy" in metrics
        assert "macro_f1" in metrics
        print("  -> Passed!")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_inference_and_gradcam():
    print("[Test 5] Testing inference and Grad-CAM generation...")
    sample_img = list(IMAGES_DIR.glob("*/*.jpg"))[0]
    predictor = DogBreedPredictor(model_name="efficientnetv2_b0")
    result = predictor.predict(sample_img, top_k=5, return_gradcam=True)

    assert "top_breed" in result
    assert "top_confidence" in result
    assert len(result["predictions"]) == 5
    assert "gradcam_overlay" in result
    assert isinstance(result["gradcam_overlay"], Image.Image)
    assert isinstance(result["gradcam_heatmap"], Image.Image)
    print("  -> Passed!")


def test_gradio_app():
    print("[Test 6] Testing Gradio app construction...")
    app = build_gradio_app()
    assert app is not None
    print("  -> Passed!")
def test_onnx_export_and_runtime():
    print("[Test 7] Testing ONNX export and CoreML/CPU ONNX runtime...")
    from dog_breed_classification.export_onnx import export_to_onnx
    from dog_breed_classification.predict_onnx import ONNXDogBreedPredictor

    temp_dir = Path(tempfile.mkdtemp())
    try:
        onnx_file = temp_dir / "test_model.onnx"
        export_to_onnx(
            model_name="mobilenetv3_large",
            output_path=onnx_file,
            verbose=False,
        )
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 1000

        predictor = ONNXDogBreedPredictor(onnx_path=onnx_file)
        sample_img = list(IMAGES_DIR.glob("*/*.jpg"))[0]
        res = predictor.predict(sample_img, top_k=5)

        assert "top_breed" in res
        assert "top_confidence" in res
        assert len(res["predictions"]) == 5
        assert res["runtime"] == "ONNXRuntime"
        print(f"  -> Passed! (Providers: {res['providers']})")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    print("=" * 60)
    print("Running Complete End-to-End Pipeline Verification")
    print("=" * 60)
    test_clean_breed_name()
    test_dataset_indexing()
    test_model_architectures()
    test_micro_training_and_evaluation()
    test_inference_and_gradcam()
    test_gradio_app()
    test_onnx_export_and_runtime()
    print("=" * 60)
    print("ALL TESTS (INCLUDING ONNX) PASSED WITH 100% SUCCESS!")
    print("=" * 60)


if __name__ == "__main__":
    main()
