# 🐕 Dog Breed Classification & Visual Explainability (Grad-CAM)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Metal%20GPU-000000?style=flat&logo=apple&logoColor=white)](https://developer.apple.com/metal/)
[![Dataset](https://img.shields.io/badge/Dataset-Stanford%20Dogs%20(120%20Breeds)-blue)](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange?style=flat&logo=gradio&logoColor=white)](https://gradio.app/)
[![ONNX](https://img.shields.io/badge/Format-ONNX%20%2B%20CoreML-005CED?style=flat&logo=onnx&logoColor=white)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
An end-to-end deep learning system for fine-grained image classification across all **120 dog breeds** in the **Stanford Dogs Dataset** (20,580 images). Optimized natively for **macOS Apple Silicon (M-series GPUs via `tensorflow-metal` and CoreML via `onnxruntime`)** with a two-stage transfer learning pipeline, **Grad-CAM** visual explainability, an interactive **Gradio Web App**, a full-featured **CLI**, and a comprehensive **Jupyter Notebook**.

---

## 🌟 Key Features

- **🚀 Native Apple Silicon Acceleration**: High-throughput GPU training and inference on Apple M1/M2/M3/M4 chips via `tensorflow-metal`.
- **🧠 Advanced Transfer Learning Backbones**: Pre-trained support for `EfficientNetV2-S`, `EfficientNetV2-B0`, `MobileNetV3-Large`, and `ResNet50V2`.
- **🔄 Two-Stage Fine-Tuning Pipeline**:
  - *Phase 1 (Feature Extraction)*: Frozen backbone with a custom regularized classification head (Dropout, BatchNorm, L2 Regularization, Label Smoothing).
  - *Phase 2 (Fine-Tuning)*: Selective unfreezing of upper convolutional layers with learning rate scheduling and Early Stopping.
- **⚡ High-Performance `tf.data` Pipeline**: Asynchronous data loading, on-the-fly GPU-accelerated augmentations (flips, rotations, zooms, translations, contrast), memory caching, and prefetching.
- **🔍 Explainable AI (Grad-CAM)**: Generates visual class activation heatmaps highlighting the exact anatomical features (muzzle, ears, coat texture, facial markings) driving the model's classification.
- **🖥️ Interactive Gradio Web App**: Local web interface for drag-and-drop dog photo classification, Top-K probability charts, customizable Grad-CAM overlays, breed facts, and execution engine toggles (TensorFlow Metal vs ONNX CoreML).
- **📦 Cross-Platform ONNX Export & CoreML Runtime**: One-command export to `.onnx` format with zero-dependency inference on macOS via `CoreMLExecutionProvider` (Apple Neural Engine/GPU) and CPU.
- **🛠️ Rich CLI Interface**: Single command-line entry point (`dog-breed`) for training, evaluation, single-image inference, ONNX export, and diagnostics.
---

## 📁 Repository Structure

```text
dog-breed-classification/
├── data/
│   ├── images/Images/              # 120 breed subfolders (20,580 images)
│   └── annotations/Annotation/     # Pascal VOC bounding box XML annotations
├── src/
│   └── dog_breed_classification/
│       ├── __init__.py             # Package initializer
│       ├── config.py               # Paths, hyperparameters, and Apple Metal GPU setup
│       ├── dataset.py              # Dataset loader, name cleaner, splits, and tf.data
│       ├── models.py               # EfficientNetV2/MobileNetV3 backbones and heads
│       ├── train.py                # Two-phase training engine and callbacks
│       ├── evaluate.py             # Top-1/Top-5 accuracy, reports, confusion plots
│       ├── explainability.py       # Grad-CAM heatmap engine and overlay blending
│       ├── predict.py              # Single/batch inference and predictor class
│       ├── export_onnx.py          # Keras to ONNX model export converter
│       ├── predict_onnx.py         # High-throughput ONNX Runtime predictor
│       ├── app.py                  # Interactive Gradio web application
│       └── cli.py                  # Full-featured command-line interface
│   └── test_pipeline.py            # End-to-end integration and smoke test suite
├── artifacts/                      # Model weights, logs, reports, and plots
│   ├── models/                     # Saved .keras models and class_names.json
│   ├── checkpoints/                # Phase 1 & 2 checkpoint weights
│   ├── logs/                       # TensorBoard events and CSVLogger histories
│   ├── plots/                      # Training curves and Grad-CAM visualizations
│   └── reports/                    # Evaluation JSON summaries and classification CSVs
├── dog_breed_classification.ipynb  # Interactive walkthrough Jupyter Notebook
├── pyproject.toml                  # Python package configuration and dependencies
└── README.md                       # Documentation and usage guide
```

---

## ⚡ Quickstart & Installation (macOS)

### 1. Prerequisites
Ensure you have Python 3.11+ installed on your Mac. Using [`uv`](https://github.com/astral-sh/uv) is recommended for blazing fast environment resolution.

```bash
# Clone repository
git clone https://github.com/AtulPahal/dog-breed-classification.git
cd dog-breed-classification

# Install dependencies with uv
uv sync
```

### 2. Verify Apple Silicon Metal GPU Support
Check that TensorFlow detects your Apple Silicon GPU:

```bash
uv run dog-breed info
```

Output:
```text
╭──────────────────────────────────────────────────────────────────────────────╮
│ 🐕 Stanford Dogs Breed Classification                                        │
│ Fine-Grained Classification with Transfer Learning & Apple Silicon Metal GPU │
╰──────────────────────────────────────────────────────────────────────────────╯
              💻 System & Hardware Environment              
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Property            ┃ Value                              ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Python Version      │ 3.11.15                            │
│ TensorFlow Version  │ 2.16.2                             │
│ Compute Accelerator │ Apple Silicon Metal GPU (1 device) │
│ GPU [0]             │ /physical_device:GPU:0             │
└─────────────────────┴────────────────────────────────────┘
    📊 Stanford Dogs Dataset Summary     
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Metric                ┃ Count / Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Total Images          │ 20,580        │
│ Target Dog Breeds     │ 120           │
│ Min Images per Breed  │ 148           │
│ Max Images per Breed  │ 252           │
│ Mean Images per Breed │ 171.5         │
└───────────────────────┴───────────────┘
```

---

## 💻 Usage Guide

### 1. Launch Interactive Web UI (Gradio)
Experience instant predictions, top-5 probability rankings, and live Grad-CAM heatmaps:

```bash
uv run dog-breed app
```
Navigate to `http://127.0.0.1:7860` in your web browser.

---

### 2. Classify an Image via CLI
Run prediction on any dog image file and export the Grad-CAM attention heatmap:

```bash
uv run dog-breed predict data/images/Images/n02085620-Chihuahua/n02085620_10074.jpg \
  --model-name efficientnetv2_s \
  --top-k 5 \
  --save-cam artifacts/plots/chihuahua_gradcam.png
```

---

### 3. Train a Model
Train a complete model from scratch using two-stage transfer learning:

```bash
uv run dog-breed train \
  --model-name efficientnetv2_s \
  --initial-epochs 10 \
  --fine-tune-epochs 15 \
  --batch-size 32 \
  --lr 0.001 \
  --fine-tune-lr 0.0001 \
  --image-size 224
```

Supported backbones:
- `efficientnetv2_s` (Default — best accuracy/speed balance)
- `efficientnetv2_b0` (Lightweight and fast)
- `mobilenetv3_large` (Ultra-fast mobile architecture)
- `resnet50v2` (Classic benchmark)

---

### 4. Evaluate Model on Test Split
Compute quantitative metrics including Top-1 accuracy, Top-5 accuracy, classification reports, and confusion analysis:

```bash
uv run dog-breed evaluate --model-name efficientnetv2_s --batch-size 32
```

---

### 5. Interactive Jupyter Notebook
Launch Jupyter to explore the step-by-step walkthrough notebook:

```bash
uv run jupyter lab dog_breed_classification.ipynb
```

The notebook covers:
1. Hardware & Environment diagnostics
2. Dataset exploration & bounding box visualization
3. Data augmentation visualizer
4. Transfer learning architecture build & inspection
5. Phase 1 & Phase 2 training execution
6. Test set evaluation & Top Confused Dog Breeds
7. Grad-CAM visual attention mapping across diverse breeds


---

### 6. Export to ONNX & Run with CoreML Acceleration
Export your trained model to standard `.onnx` format:

```bash
uv run dog-breed export-onnx --model-name efficientnetv2_s --output artifacts/models/dog_classifier_efficientnetv2_s.onnx
```

Perform high-throughput, framework-agnostic prediction using `onnxruntime` with Apple Silicon CoreML hardware acceleration:

```bash
uv run dog-breed predict data/images/Images/n02085620-Chihuahua/n02085620_10074.jpg \
  --model-path artifacts/models/dog_classifier_efficientnetv2_s.onnx \
  --onnx \
  --top-k 5
```
---

## 🧪 Testing & Verification

Run the end-to-end integration and smoke test suite:

```bash
uv run python tests/test_pipeline.py
```

All 6 test stages verify dataset indexing, data augmentation, model compilation, unfreezing, micro-training on Metal GPU, inference, Grad-CAM, and Gradio app construction.

---

## 📊 Methodology & Technical Details

### 1. Data Augmentation
Stochastic transformations are applied on-the-fly to batches of tensors:
- Random Horizontal Flips
- Random Rotations ($\pm 12\%$)
- Random Zoom ($\pm 12\%$)
- Random Translations ($\pm 8\%$)
- Random Contrast adjustments ($\pm 10\%$)

### 2. Regularization & Loss
- **Label Smoothing ($\epsilon = 0.1$)**: Mitigates overconfidence in fine-grained inter-breed visual overlap.
- **L2 Weight Regularization ($10^{-4}$)**: Penalizes large weights in dense classification layers.
- **Dropout ($p = 0.3$)**: Prevents co-adaptation of neurons in classification head.
- **AdamW / Adam Optimizer**: With ReduceLROnPlateau and EarlyStopping callbacks.

### 3. Grad-CAM Explainability Formulation
Grad-CAM computes the weight of each convolutional activation map $k$ for class $c$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

---

## 📜 Dataset Reference & Citation

- **Stanford Dogs Dataset**: Aditya Khosla, Nityananda Jayadevaprakash, Bangpeng Yao, and Li Fei-Fei. *Novel Dataset for Fine-Grained Image Categorization*. First Workshop on Fine-Grained Visual Categorization (FGVC), IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.
- **Dataset Link**: [Kaggle - Stanford Dogs Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset)


## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
---

## 👨‍💻 Author

- **Atul Pahal** ([GitHub](https://github.com/AtulPahal))
