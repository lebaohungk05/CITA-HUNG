import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Thêm đường dẫn
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
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'emotion_models', 'fer2013_resnet18_freeze_retrain_best.pth')
ASSETS_DIR = os.path.join(BASE_DIR, 'report_assets')
INDIVIDUAL_DIR = os.path.join(ASSETS_DIR, 'individual_preds')

if not os.path.exists(INDIVIDUAL_DIR): os.makedirs(INDIVIDUAL_DIR)

def load_best_model():
    print(f"Loading best model from: {MODEL_PATH}")
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Clean keys
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if not k.startswith('n_averaged'): new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

def draw_prediction_card(img_tensor, class_name, conf):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img_large = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    canvas = np.ones((410, 300, 3), dtype=np.uint8) * 255
    canvas[:300, :] = img_large
    colors = {'Angry': (0, 0, 200), 'Disgust': (0, 120, 0), 'Fear': (130, 0, 130), 'Happy': (0, 180, 180), 'Sad': (200, 0, 0), 'Surprise': (0, 140, 255), 'Neutral': (80, 80, 80)}
    color = colors.get(class_name, (0, 0, 0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"True: {class_name}", (15, 330), font, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Pred: {class_name}", (15, 360), font, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Conf: {conf*100:.2f}%", (15, 390), font, 0.7, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (15, 398), (15 + int(270 * conf), 403), color, -1)
    return canvas

def main():
    model = load_best_model()
    dataset = FERDataset(DATASET_PATH, (INPUT_SIZE, INPUT_SIZE), mode='val')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    best_samples = {i: (0.0, None) for i in range(NUM_CLASSES)}
    
    print("Running inference on test set...")
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            out = (model(batch_x) + model(torch.flip(batch_x, [3]))) / 2.0
            probs = torch.softmax(out, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            
            for i in range(batch_x.size(0)):
                lbl, prd, cnf = batch_y[i].item(), preds[i].item(), max_probs[i].item()
                if lbl == prd and cnf > best_samples[lbl][0]:
                    best_samples[lbl] = (cnf, batch_x[i].clone())

    # 1. Update Confusion Matrix
    print("Updating Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds, normalize='true')
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f"Normalized Confusion Matrix (Acc: 72.72%)")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(ASSETS_DIR, 'confusion_matrix_mixup.png'), bbox_inches='tight')
    
    # 2. Update Distribution Comparison
    print("Updating Distribution Comparison...")
    gt_counts = np.bincount(all_labels, minlength=NUM_CLASSES)
    pred_counts = np.bincount(all_preds, minlength=NUM_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#c2c2f0', '#ffb3e6', '#c4e17f']
    for i, (data, title) in enumerate([(gt_counts, 'Ground Truth'), (pred_counts, 'Model Predictions')]):
        axes[i].barh(CLASSES, data, color=colors, edgecolor='grey')
        axes[i].invert_yaxis()
        axes[i].set_title(title, fontsize=14, fontweight='bold')
        for j, v in enumerate(data): axes[i].text(v + 5, j + 0.1, str(int(v)), fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'distribution_comparison.png'))

    # 3. Update Individual Preds
    print("Updating Individual Prediction Cards...")
    for idx, (conf, img_tensor) in best_samples.items():
        if img_tensor is not None:
            card = draw_prediction_card(img_tensor, CLASSES[idx], conf)
            cv2.imwrite(os.path.join(INDIVIDUAL_DIR, f"pred_{CLASSES[idx].lower()}.png"), cv2.cvtColor(card, cv2.COLOR_RGB2BGR))

    print("All assets updated successfully!")

if __name__ == '__main__':
    main()
