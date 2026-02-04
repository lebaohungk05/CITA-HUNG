import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# IMPORT MODEL CUSTOM MỚI
from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

# --- Cấu hình Mixup & Custom (TINH CHỈNH CHO MỤC TIÊU 73%) ---
BATCH_SIZE = 128
NUM_EPOCHS = 280      # Tăng từ 240 -> 280 để model có thời gian ngấm bài toán khó hơn
SWA_START_EPOCH = 230 # Tăng từ 200 -> 230 để Cosine Annealing chạy lâu hơn
INPUT_SIZE = 48
NUM_CLASSES = 7
MIXUP_ALPHA = 0.6     # Tăng từ 0.4 -> 0.6: Tăng độ khó, buộc model học đặc trưng tổng quát

# Class Weights (Giữ nguyên)
CLASS_WEIGHTS = [1.02, 3.10, 1.01, 0.76, 0.93, 1.15, 0.91]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, '../trained_models/emotion_models/')
DATASET_PATH = os.path.join(SCRIPT_DIR, '../datasets')

# Đường dẫn riêng cho bản Mixup Optimized (Để không ghi đè bản cũ 72.5%)
TENSORBOARD_LOG_DIR = os.path.join(BASE_PATH, 'tensorboard_logs_resnet_mixup_optimized')
CSV_LOG_PATH = os.path.join(BASE_PATH, 'fer2013_resnet_training_log_mixup_optimized.csv')
CHECKPOINT_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_mixup_optimized.pth')
BEST_MODEL_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_best_optimized.pth')

# --- Mixup Functions ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

if __name__ == "__main__":
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | MIXUP OPTIMIZED (Alpha={MIXUP_ALPHA}, WD=2e-4)")

    # 2. Prepare Paths
    if not os.path.exists(BASE_PATH): os.makedirs(BASE_PATH)

    # 3. Datasets
    print("Initializing Datasets...")
 
    train_dataset = FERDataset(os.path.join(DATASET_PATH, 'train'), (INPUT_SIZE, INPUT_SIZE), mode='train')
    test_dataset = FERDataset(os.path.join(DATASET_PATH, 'test'), (INPUT_SIZE, INPUT_SIZE), mode='val')

    print(f"--> Train set size: {len(train_dataset)}")
    print(f"--> Test set size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model Initialization (CUSTOM MODEL)
    print("Building Model (Custom ResNet18 for 48x48)...")
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=True)
    model.to(device)
    
    # SWA Model Wrapper
    swa_model = AveragedModel(model)
    swa_model.to(device)

    # 5. Optimizer & Loss
    class_weights_tensor = torch.FloatTensor(CLASS_WEIGHTS).to(device)
    # Giữ Label Smoothing 0.1
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    
    # Tăng Weight Decay lên 2e-4 để kháng overfitting khi train lâu hơn
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=2e-4)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=SWA_START_EPOCH)
    # Giảm swa_lr xuống 0.001 (1e-3) để tránh shock loss khi chuyển phase
    swa_scheduler = SWALR(optimizer, swa_lr=0.001)

    # 6. Logging & Checkpoint Loading
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    
    start_epoch = 1
    best_acc = 0.0
    
    if start_epoch == 1 and os.path.exists(CSV_LOG_PATH):
        try:
            os.remove(CSV_LOG_PATH)
            print("Removed old log file.")
        except: pass

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'swa_model' in checkpoint:
                swa_model.load_state_dict(checkpoint['swa_model'])
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            print(f"Resumed from Epoch {start_epoch}, Best Acc: {best_acc:.2f}%")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")

    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr', 'phase'])

    # 7. Training Loop
    print(f"Training Started. Total Epochs: {NUM_EPOCHS}. Mixup Alpha: {MIXUP_ALPHA}")
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        is_swa_phase = epoch >= SWA_START_EPOCH
        phase_name = "SWA" if is_swa_phase else "STD"
        
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # Mixup Execution
            mixed_x, y_a, y_b, lam = mixup_data(batch_x, batch_y, MIXUP_ALPHA, device)
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += batch_y.size(0)
            # Tính accuracy dựa trên nhãn mạnh hơn (label dominant)
            correct += (lam * (pred == y_a).sum().item() + (1 - lam) * (pred == y_b).sum().item())

        # SWA Update & Scheduler Step
        if is_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        # Validation (TTA)
        eval_model = model
        if is_swa_phase:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            eval_model = swa_model
        
        eval_model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                # TTA: Flip
                outputs1 = eval_model(batch_x)
                outputs2 = eval_model(torch.flip(batch_x, [3]))
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

        print(f"Epoch [{epoch}/{NUM_EPOCHS}] ({phase_name}) | Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} | Acc: {train_acc:.2f}%/{val_acc:.2f}% | LR: {current_lr:.4f}")

        # Logging
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Acc/Train', train_acc, epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)
        
        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_train_loss, avg_val_loss, train_acc, val_acc, current_lr, phase_name])

        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(eval_model.state_dict(), BEST_MODEL_PATH)
            print(f"  --> New Best Acc: {best_acc:.2f}%")
        
        # Save Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'swa_model': swa_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, CHECKPOINT_PATH)

    print(f"Training Finished. Final Best Accuracy: {best_acc:.2f}%")
    writer.close()