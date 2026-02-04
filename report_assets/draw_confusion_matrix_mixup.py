import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import itertools
from sklearn.metrics import confusion_matrix

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # face_classification
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12, fontweight='bold')

    plt.ylabel('True Label', fontsize=14, labelpad=10)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.tight_layout()

def generate_confusion_matrix():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    BATCH_SIZE = 128
    INPUT_SIZE = 48
    # Absolute path calculation
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '../datasets/test')
    model_path = os.path.join(base_dir, '../trained_models/emotion_models/fer2013_resnet18_mixup_best.pth')
    output_image_path = os.path.join(base_dir, 'confusion_matrix_mixup.png')

    # 2. Data Transforms are handled by FERDataset now
    
    # 3. Load Data
    print(f"Loading test dataset from {dataset_path}...")
    # Mode 'val' ensures correct validation transforms (Resize 48x48, Normalize)
    test_dataset = FERDataset(dataset_path, image_size=(INPUT_SIZE, INPUT_SIZE), mode='val')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Standard FER2013 Order from FERDataset
    class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    print(f"Classes: {class_names}")

    # 4. Load Model
    print("Loading model...")
    model = resnet18_custom(num_classes=7, pretrained=False)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        # Handle if checkpoint wraps state_dict (which it does in train_resnet_mixup.py)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_acc = checkpoint.get('best_acc', 'N/A')
            print(f"Loaded checkpoint with Best Acc: {best_acc}%")
        else:
            model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    else:
        print(f"Error: Model path not found: {model_path}")
        return

    model.to(device)
    model.eval()

    # 5. Inference
    y_true = []
    y_pred = []
    
    print("Running inference with TTA (Horizontal Flip)...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            
            # TTA: Forward pass with original images
            outputs1 = model(inputs)
            
            # TTA: Forward pass with flipped images
            outputs2 = model(torch.flip(inputs, [3])) # Flip along width dimension
            
            # Average predictions
            outputs_avg = (outputs1 + outputs2) / 2.0
            
            _, preds = torch.max(outputs_avg, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # 6. Calculate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy from CM
    acc = np.trace(cm) / np.sum(cm) * 100
    print(f"Computed Accuracy on Test Set: {acc:.2f}%")

    # 7. Plot and Save
    print("Plotting...")
    plot_confusion_matrix(cm, classes=class_names, normalize=True, 
                          title=f'Confusion Matrix (Acc: {acc:.2f}%)')
    
    # Saving with optimized size (DPI 90 similar to pipeline fig)
    plt.savefig(output_image_path, dpi=90, bbox_inches='tight')
    print(f"Confusion Matrix saved to: {output_image_path}")
    
    # Also save raw text numbers for reference
    text_path = output_image_path.replace('.png', '.txt')
    with open(text_path, 'w') as f:
        f.write(f"Accuracy: {acc:.2f}%\n")
        f.write(np.array2string(cm))
    print(f"Raw data saved to: {text_path}")

if __name__ == "__main__":
    generate_confusion_matrix()
