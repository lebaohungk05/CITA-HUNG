import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from tqdm import tqdm

# Import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

# --- CẤU HÌNH ---
BATCH_SIZE = 64
INPUT_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Danh sách các model cần Ensemble
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_PATH, 'trained_models', 'emotion_models')

MODEL_PATHS = [
    os.path.join(MODEL_DIR, 'fer2013_resnet18_freeze_retrain_best.pth'), # 72.72%
    os.path.join(MODEL_DIR, 'fer2013_resnet18_best_optimized.pth'),      # 72.54%
    os.path.join(MODEL_DIR, 'fer2013_resnet18_mixup_best.pth')           # 72.50%
]

def load_model(path):
    if not os.path.exists(path):
        print(f"Warning: Model not found at {path}")
        return None
    
    print(f"Loading: {os.path.basename(path)}")
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(path, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Clean state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if not k.startswith('n_averaged'): 
            new_state_dict[name] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def main():
    print(f"--- ENSEMBLE EVALUATION ({len(MODEL_PATHS)} Models) ---")
    
    # 1. Load Data
    dataset_path = os.path.join(BASE_PATH, 'datasets', 'test')
    test_dataset = FERDataset(dataset_path, (INPUT_SIZE, INPUT_SIZE), mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Test set size: {len(test_dataset)}")

    # 2. Load Models
    models = []
    for path in MODEL_PATHS:
        m = load_model(path)
        if m: models.append(m)
    
    if not models:
        print("No models loaded!")
        return

    # 3. Inference
    correct = 0
    total = 0
    
    print("Running Ensemble Inference...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            avg_probs = None
            
            # Duyệt qua từng model
            for model in models:
                # TTA Flip cho từng model luôn (Tăng độ chính xác tối đa)
                out1 = model(batch_x)
                out2 = model(torch.flip(batch_x, [3]))
                output = (out1 + out2) / 2.0
                
                probs = torch.softmax(output, dim=1)
                
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs += probs
            
            # Chia trung bình
            avg_probs /= len(models)
            
            # Lấy nhãn
            _, preds = torch.max(avg_probs, 1)
            
            total += batch_y.size(0)
            correct += (preds == batch_y).sum().item()

    acc = 100 * correct / total
    print(f"\n------------------------------------------------")
    print(f"ENSEMBLE ACCURACY: [92m{acc:.4f}%[0m")
    print(f"------------------------------------------------")

if __name__ == "__main__":
    main()
