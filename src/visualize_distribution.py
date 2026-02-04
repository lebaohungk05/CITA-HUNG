import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') # Set non-GUI backend
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Thêm đường dẫn để import được modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

# Cấu hình
BATCH_SIZE = 64
INPUT_SIZE = 48
NUM_CLASSES = 7
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'test')
# Sử dụng model Mixup Best cũ (72.5%) để visualize
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'emotion_models', 'fer2013_resnet18_mixup_best.pth')
OUTPUT_IMAGE_PATH = os.path.join(BASE_DIR, 'report_assets', 'distribution_comparison.png')

def main():
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = FERDataset(DATASET_PATH, (INPUT_SIZE, INPUT_SIZE), mode='val')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 1. Đếm số lượng Ground Truth
    gt_counts = np.zeros(NUM_CLASSES)
    # Lấy label trực tiếp từ dataset (nhanh hơn loop qua loader)
    # FERDataset lưu danh sách file dưới dạng (path, label_idx)
    for _, label in dataset.samples:
        gt_counts[label] += 1
        
    print(f"Ground Truth Counts: {gt_counts}")

    # 2. Load Model & Dự đoán
    print(f"Loading model from {MODEL_PATH}...")
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Xử lý key nếu model được lưu khác format
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Model file not found! Using random weights (Just for demo structure).")
        
    model.to(DEVICE)
    model.eval()

    pred_counts = np.zeros(NUM_CLASSES)
    print("Running inference...")
    
    with torch.no_grad():
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(DEVICE)
            outputs = model(batch_x)
            
            # TTA lật ảnh để dự đoán chính xác hơn (giống lúc training)
            outputs_flip = model(torch.flip(batch_x, [3]))
            outputs = (outputs + outputs_flip) / 2.0
            
            _, preds = torch.max(outputs, 1)
            
            for p in preds.cpu().numpy():
                pred_counts[p] += 1

    print(f"Predicted Counts: {pred_counts}")

    # 3. Vẽ biểu đồ
    print("Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Màu sắc
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0', '#ffb3e6', '#c4e17f']

    # Chart 1: Ground Truth
    y_pos = np.arange(len(CLASSES))
    axes[0].barh(y_pos, gt_counts, color=colors, edgecolor='grey')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(CLASSES, fontsize=12)
    axes[0].invert_yaxis()  # Đảo ngược để Angry ở trên cùng
    axes[0].set_xlabel('Number of Images', fontsize=12)
    axes[0].set_title('Original Dataset Distribution (Test Set)', fontsize=14, fontweight='bold')
    
    # Thêm số liệu lên thanh
    for i, v in enumerate(gt_counts):
        axes[0].text(v + 5, i + 0.1, str(int(v)), color='black', fontweight='bold')

    # Chart 2: Model Predictions
    axes[1].barh(y_pos, pred_counts, color=colors, edgecolor='grey')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(CLASSES, fontsize=12)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Number of Predictions', fontsize=12)
    axes[1].set_title(f'Model Prediction Distribution\n(ResNet Mixup - ~72.5% Acc)', fontsize=14, fontweight='bold')

    for i, v in enumerate(pred_counts):
        axes[1].text(v + 5, i + 0.1, str(int(v)), color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_PATH)
    print(f"Chart saved to {OUTPUT_IMAGE_PATH}")

if __name__ == '__main__':
    main()
