import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
import sys

# Thêm đường dẫn
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

# Cấu hình
BATCH_SIZE = 32
INPUT_SIZE = 48
NUM_CLASSES = 7
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'datasets', 'test')
# Sử dụng model Mixup Best (72.5%)
MODEL_PATH = os.path.join(BASE_DIR, 'trained_models', 'emotion_models', 'fer2013_resnet18_mixup_best.pth')
OUTPUT_DIR = os.path.join(BASE_DIR, 'report_assets', 'individual_preds')

def draw_prediction_card(img_tensor, label_idx, conf, class_name):
    """
    Vẽ một 'card' chuyên nghiệp cho ảnh dự đoán:
    - Ảnh gốc được upscale
    - Hiển thị: True Label, Pred Label, Confidence
    """
    # 1. Denormalize & Convert to CV2 Image
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    
    # Upscale ảnh (48x48 -> 300x300)
    img_large = cv2.resize(img, (300, 300), interpolation=cv2.INTER_LINEAR)
    
    # Tạo canvas rộng hơn một chút để chứa 3 dòng chữ (300x400)
    canvas = np.ones((410, 300, 3), dtype=np.uint8) * 255
    canvas[:300, :] = img_large
    
    # Màu sắc
    colors = {
        'Angry': (0, 0, 200), 'Disgust': (0, 120, 0), 'Fear': (130, 0, 130),
        'Happy': (0, 180, 180), 'Sad': (200, 0, 0), 'Surprise': (0, 140, 255),
        'Neutral': (80, 80, 80)
    }
    color = colors.get(class_name, (0, 0, 0))

    # Vẽ Text chi tiết
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"True: {class_name}", (15, 330), font, 0.7, (50, 50, 50), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Pred: {class_name}", (15, 360), font, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Conf: {conf*100:.2f}%", (15, 390), font, 0.7, (50, 50, 50), 1, cv2.LINE_AA)
    
    # Vẽ thanh Confidence bar ở dưới cùng
    bar_width = int(270 * conf)
    cv2.rectangle(canvas, (15, 398), (15 + bar_width, 403), color, -1)
    
    return canvas

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = FERDataset(DATASET_PATH, (INPUT_SIZE, INPUT_SIZE), mode='val')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Loading model from {MODEL_PATH}...")
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(DEVICE)
    model.eval()

    # Dictionary lưu mẫu tốt nhất cho mỗi lớp: {class_idx: (max_conf, image_tensor)}
    best_samples = {i: (0.0, None) for i in range(NUM_CLASSES)}
    found_count = 0

    print("Searching for high-confidence correct predictions...")
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            outputs = model(batch_x)
            # TTA Flip
            outputs_flip = model(torch.flip(batch_x, [3]))
            outputs = (outputs + outputs_flip) / 2.0
            
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            # Duyệt qua từng ảnh trong batch
            for i in range(batch_x.size(0)):
                label = batch_y[i].item()
                pred = preds[i].item()
                conf = max_probs[i].item()
                
                # Chỉ lấy ảnh dự đoán ĐÚNG
                if label == pred:
                    # Nếu độ tin cậy cao hơn mẫu cũ -> Cập nhật
                    if conf > best_samples[label][0]:
                        best_samples[label] = (conf, batch_x[i].clone())

    print("Saving individual images...")
    for idx, (conf, img_tensor) in best_samples.items():
        if img_tensor is not None:
            class_name = CLASSES[idx]
            card = draw_prediction_card(img_tensor, idx, conf, class_name)
            
            filename = f"pred_{class_name.lower()}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            # Convert RGB to BGR for OpenCV saving
            card_bgr = cv2.cvtColor(card, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, card_bgr)
            print(f"  -> Saved {filename} (Conf: {conf:.4f})")
        else:
            print(f"  -> Warning: No correct prediction found for class {CLASSES[idx]}")

    print(f"\nDone! Images saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
