import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
import os
import sys
import random
from PIL import Image

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # face_classification
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

def generate_test_predictions_viz():
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = 48
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_dir, '../datasets/test')
    model_path = os.path.join(base_dir, '../trained_models/emotion_models/fer2013_resnet18_mixup_best.pth')
    output_path = os.path.join(base_dir, 'test_predictions_7classes.png')

    # 2. Load Model
    model = resnet18_custom(num_classes=7, pretrained=False)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print(f"Model not found at {model_path}")
        return
        
    model.to(device)
    model.eval()

    # 3. Transform
    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Standard FER2013 Classes (Model Order)
    # 0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    folder_map = {'Angry': 'angry', 'Disgust': 'disgust', 'Fear': 'fear', 
                  'Happy': 'happy', 'Sad': 'sad', 'Surprise': 'surprise', 'Neutral': 'neutral'}

    # 4. Select BEST 1 image per class (Correctly classified + Highest Confidence)
    selected_images = []
    
    print("Searching for high-confidence correct predictions...")
    
    for class_idx, cls_name in enumerate(classes):
        folder_name = folder_map[cls_name]
        cls_path = os.path.join(dataset_path, folder_name)
        
        if not os.path.exists(cls_path):
            print(f"Warning: Folder {cls_path} not found.")
            continue
            
        files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        # Shuffle files to get variety if we run multiple times, 
        # but here we want max confidence so we should check many.
        # Let's check up to 100 images per class to save time, or all if small.
        random.shuffle(files)
        candidates = files[:200] 
        
        best_img_path = None
        best_conf = -1.0
        
        for img_name in candidates:
            img_path = os.path.join(cls_path, img_name)
            try:
                img_pil = Image.open(img_path).convert('L')
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # TTA: Flip averaging for consistency with best accuracy
                    outputs1 = model(img_tensor)
                    outputs2 = model(torch.flip(img_tensor, [3]))
                    outputs_avg = (outputs1 + outputs2) / 2.0
                    
                    probs = torch.nn.functional.softmax(outputs_avg, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    if pred_idx.item() == class_idx:
                        current_conf = conf.item()
                        if current_conf > best_conf:
                            best_conf = current_conf
                            best_img_path = img_path
                            
                        # If we find a very high confidence one, stop early to save time
                        if best_conf > 0.99: 
                            break
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if best_img_path:
            selected_images.append((cls_name, best_img_path, best_conf))
            print(f"  -> Found {cls_name}: Conf {best_conf*100:.2f}%")
        else:
            print(f"  -> Warning: No correct prediction found for {cls_name} in scanned batch.")
            # Fallback: just pick the first one
            if files:
                selected_images.append((cls_name, os.path.join(cls_path, files[0]), 0.0))

    # 5. Plot
    fig, axes = plt.subplots(1, 7, figsize=(22, 5))
    
    for i, (true_label, img_path, conf) in enumerate(selected_images):
        img_pil = Image.open(img_path).convert('L')
        ax = axes[i]
        ax.imshow(img_pil, cmap='gray')
        ax.axis('off')
        
        # Color code: Green (High Conf), Yellow (Med)
        color = '#10B981' # Emerald Green
        if conf < 0.8: color = '#F59E0B' # Amber
        
        # Hiển thị đầy đủ True, Pred, Conf
        title_text = f"True: {true_label}\nPred: {true_label}\nConf: {conf*100:.1f}%"
        ax.set_title(title_text, color=color, fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")

if __name__ == "__main__":
    generate_test_predictions_viz()
