import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

# Settings
BATCH_SIZE = 64
INPUT_SIZE = 48
NUM_CLASSES = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'emotion_models', 'fer2013_resnet18_mixup_best.pth')
DATASET_DIR = os.path.join(BASE_DIR, 'datasets', 'test')

def evaluate_model():
    print(f"Evaluation Script for: {MODEL_PATH}")
    print(f"Dataset: {DATASET_DIR}")
    
    # 1. Load Data
    test_dataset = FERDataset(DATASET_DIR, (INPUT_SIZE, INPUT_SIZE), mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # 2. Load Model
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # Handle state dict wrapper
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(DEVICE)
    model.eval()

    # 3. Inference
    all_preds = []
    all_labels = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # TTA (Test Time Augmentation) - Enabled to match best reported accuracy
            # Original
            outputs1 = model(inputs)
            # Flip
            outputs2 = model(torch.flip(inputs, [3]))
            outputs = (outputs1 + outputs2) / 2.0
            
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    # Mapping based on FERDataset class_to_idx: 
    # {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    # Note: Ensure this order matches exactly what FERDataset uses. 
    # FERDataset uses: {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    # Wait, 'sad' is index 4, 'surprise' is 5? Standard FER2013 usually has 7 classes.
    # Let's trust FERDataset's internal dict.
    
    # Re-verify FERDataset mapping from datasets.py read earlier:
    # self.class_to_idx = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    # Yes, that matches.
    
    # Align class_names list to indices 0..6
    ordered_class_names = [None] * 7
    mapping = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4, 'surprise': 5, 'neutral': 6}
    for name, idx in mapping.items():
        ordered_class_names[idx] = name
        
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=ordered_class_names, digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

if __name__ == "__main__":
    evaluate_model()
