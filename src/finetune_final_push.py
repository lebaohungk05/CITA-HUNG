import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import model
from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

# --- CẤU HÌNH FINAL PUSH ---
BATCH_SIZE = 128
NUM_EPOCHS = 30           # Chạy ngắn
INPUT_SIZE = 48
NUM_CLASSES = 7
LEARNING_RATE = 1e-4      # LR rất nhỏ để tinh chỉnh
WEIGHT_DECAY = 5e-4       # Tăng nhẹ WD để tránh overfit khi tắt Mixup

# Class Weights
CLASS_WEIGHTS = [1.02, 3.10, 1.01, 0.76, 0.93, 1.15, 0.91]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, '../trained_models/emotion_models/')
DATASET_PATH = os.path.join(SCRIPT_DIR, '../datasets')

# Paths - LOAD TỪ CHECKPOINT OPTIMIZED VỪA TRAIN
PRETRAINED_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_best_optimized.pth')

# Output Paths mới
TENSORBOARD_LOG_DIR = os.path.join(BASE_PATH, 'tensorboard_logs_resnet_final_push')
CSV_LOG_PATH = os.path.join(BASE_PATH, 'fer2013_resnet_final_push.csv')
BEST_MODEL_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_final_push_best.pth')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | FINAL PUSH MODE (No Mixup, Low LR)")

    # 1. Datasets (Giữ nguyên)
    print("Initializing Datasets...")
    train_dataset = FERDataset(os.path.join(DATASET_PATH, 'train'), (INPUT_SIZE, INPUT_SIZE), mode='train')
    test_dataset = FERDataset(os.path.join(DATASET_PATH, 'test'), (INPUT_SIZE, INPUT_SIZE), mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Load Model
    print(f"Loading best checkpoint from: {PRETRAINED_PATH}")
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    
    # Load weights và xử lý lỗi tiền tố 'module.' (do lưu từ DataParallel hoặc SWA)
    checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
    
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Tạo state_dict mới bỏ tiền tố 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # Bỏ 7 ký tự đầu "module."
        elif k.startswith('n_averaged'):
            continue # Bỏ key đếm của SWA
        else:
            new_state_dict[k] = v
            
    try:
        model.load_state_dict(new_state_dict, strict=False) # strict=False để bỏ qua key thừa nếu có
        print("Model weights loaded successfully (prefix 'module.' removed).")
    except Exception as e:
        print(f"Error loading weights: {e}")
        exit(1)
        
    model.to(device)

    # 3. Optimizer (Reset Optimizer mới cho giai đoạn này)
    class_weights_tensor = torch.FloatTensor(CLASS_WEIGHTS).to(device)
    # Bỏ Label Smoothing hoặc để rất thấp (0.05) vì ta muốn model quyết đoán hơn
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.0) 
    
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 4. Logging
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    
    # Lấy accuracy hiện tại làm mốc (Mặc định 72.54% nếu checkpoint không lưu)
    best_acc = 72.54 
    if isinstance(checkpoint, dict):
        if 'best_acc' in checkpoint:
            best_acc = checkpoint['best_acc']
        elif 'val_acc' in checkpoint: # Thử key khác
            best_acc = checkpoint['val_acc']
            
    print(f"Starting Final Push. Baseline to beat: {best_acc:.2f}%")
    
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'])

    # 5. Training Loop (KHÔNG MIXUP)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Forward thường (Không Mixup)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()

        scheduler.step()

        # Validation (TTA)
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # TTA: Flip
                outputs1 = model(batch_x)
                outputs2 = model(torch.flip(batch_x, [3]))
                outputs_avg = (outputs1 + outputs2) / 2.0
                
                loss = criterion(outputs_avg, batch_y)
                val_loss += loss.item()
                _, pred = torch.max(outputs_avg, 1)
                val_total += batch_y.size(0)
                val_correct += (pred == batch_y).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        avg_val_loss = val_loss / len(test_loader)
        val_acc = 100 * val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Push Epoch [{epoch}/{NUM_EPOCHS}] | Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | Acc: {train_acc:.2f}%/{val_acc:.2f}% | LR: {current_lr:.6f}")

        # Logging
        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, train_acc, val_acc, current_lr])

        # Save Best
        if val_acc > best_acc:
            diff = val_acc - best_acc
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  --> \033[92mNEW RECORD: {best_acc:.2f}% (+{diff:.2f}%)\033[0m")
        
    print(f"Final Push Finished. Best Acc: {best_acc:.2f}%")
    writer.close()
