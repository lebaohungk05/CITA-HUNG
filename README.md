# Balancing Accuracy and Efficiency in Low-Resolution Facial Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![Framework](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) 
[![Accuracy](https://img.shields.io/badge/FER2013_Acc-72.72%25-brightgreen.svg)]()
[![Ensemble](https://img.shields.io/badge/Ensemble_Acc-73.36%25-blueviolet.svg)]()

This repository contains the official implementation of the paper: **"Balancing Accuracy and Efficiency in Low-Resolution Facial Emotion Recognition"**.

We propose **Mix-SEResNet**, a lightweight and robust framework optimized for low-resolution inputs (48x48). By combining **SE-ResNet18**, **Mixup (α=0.6)**, **Stochastic Weight Averaging (SWA)**, and a novel **Fine-tuning Strategy**, we achieve state-of-the-art results on FER2013.

<div align="center">
  <img src="report_assets/Proposed_v2_matplotlib.png" width="600px" alt="Training Strategy">
  <p><em>Figure 1: The proposed Multi-stage Training Pipeline (Convergence -> SWA -> Refinement).</em></p>
</div>

---

## 🏆 Key Results

Our method ranks **#1 among open-source methods** on FER2013 (Private Test) while maintaining extreme efficiency.

| Method | Accuracy | Parameters | Note |
| :--- | :--- | :--- | :--- |
| **Mix-SEResNet (Ensemble)** | **73.36%** | 33.9M | **Rank 1** (SOTA) |
| RM-Xception | 73.32% | 22.8M | Rank 2 |
| VGGNet | 73.28% | 138.0M | Rank 3 |
| **Mix-SEResNet (Single)** | **72.72%** | **11.3M** | **Optimal Trade-off** |
| ResNet-18 (Baseline) | 71.38% | 11.2M | |

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lebaohungk05/CITA-HUNG.git
    cd CITA-HUNG/face_classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r REQUIREMENTS.txt
    ```

3.  **Prepare Data:**
    Download FER2013 and place it in `datasets/fer2013/`.

---

## 🚀 Reproduction Steps

To reproduce our **72.72%** result, follow these 3 steps:

### Step 1: Base Training & SWA (Epoch 1-265)
Trains the model from scratch using Mixup (α=0.6) and activates SWA at epoch 230.
```bash
python src/train_resnet_mixup.py
# Output: fer2013_resnet18_best_optimized.pth (~72.54%)
```

### Step 2: Fine-tuning Refinement (Epoch 266-295)
Freezes the backbone and retrains the classifier on clean data, then unfreezes for final micro-adjustments.
```bash
python src/finetune_freeze_retrain.py
# Output: fer2013_resnet18_freeze_retrain_best.pth (72.72%)
```

### Step 3: Evaluation & Ensemble (Optional)
To achieve **73.36%**, combine the predictions of the best checkpoints.
```bash
python src/evaluate_ensemble.py
```

---

## 📊 Visualization

We provide scripts to generate all report figures:
```bash
# Generate Training History charts
python src/plot_combined_history.py

# Generate Confusion Matrix & Predictions
python src/update_all_assets.py
```

<div align="center">
  <img src="report_images/combined_accuracy_chart.png" width="45%" alt="Acc History">
  <img src="report_assets/confusion_matrix_mixup.png" width="45%" alt="Confusion Matrix">
</div>

---

## 💻 Real-time Demo

Run the webcam demo to see the model in action:
```bash
python src/video_emotion_color_demo.py
```

**Author:** Le Bao Hung