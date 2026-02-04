import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import csv
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# Import model
from models.resnet18_custom import resnet18_custom
from utils.datasets import FERDataset

# --- CẤU HÌNH ---
BATCH_SIZE = 128
INPUT_SIZE = 48
NUM_CLASSES = 7

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, '../trained_models/emotion_models/')
DATASET_PATH = os.path.join(SCRIPT_DIR, '../datasets')
PRETRAINED_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_best_optimized.pth') # Checkpoint 72.54%

# Output
CSV_LOG_PATH = os.path.join(BASE_PATH, 'fer2013_resnet_freeze_retrain.csv')
BEST_MODEL_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_freeze_retrain_best.pth')
TENSORBOARD_LOG_DIR = os.path.join(BASE_PATH, 'tensorboard_logs_resnet_freeze_retrain')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | FREEZE & RETRAIN STRATEGY")

    # TensorBoard Writer
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    # 1. Dataset & Loader (Chuẩn, không Mixup)
    print("Initializing Datasets...")
    train_dataset = FERDataset(os.path.join(DATASET_PATH, 'train'), (INPUT_SIZE, INPUT_SIZE), mode='train')
    test_dataset = FERDataset(os.path.join(DATASET_PATH, 'test'), (INPUT_SIZE, INPUT_SIZE), mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Load Model
    print(f"Loading checkpoint: {PRETRAINED_PATH}")
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    
    checkpoint = torch.load(PRETRAINED_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Fix module prefix if exists
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if not k.startswith('n_averaged'): new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)

    # Lấy best acc cũ
    best_acc = 72.54
    if isinstance(checkpoint, dict) and 'best_acc' in checkpoint:
        best_acc = checkpoint['best_acc']
    print(f"Baseline Accuracy: {best_acc:.2f}%")

    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['phase', 'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'lr'])

    # --- PHASE 1: FREEZE BACKBONE ---
    print("\n>>> PHASE 1: FREEZE BACKBONE (Train FC only) - 10 Epochs")
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
        
    optimizer = SGD(model.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss() # Không class weights, để model tự nhiên
    
    train_phase(model, train_loader, test_loader, optimizer, criterion, device, 10, best_acc, 'frozen', writer=writer)

    # --- PHASE 2: UNFREEZE ALL ---
    print("\n>>> PHASE 2: UNFREEZE ALL (Fine-tuning) - 20 Epochs")
    for param in model.parameters():
        param.requires_grad = True
        
    # LR cực nhỏ cho toàn mạng
    optimizer = SGD(model.parameters(), lr=5e-5, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    
    train_phase(model, train_loader, test_loader, optimizer, criterion, device, 20, best_acc, 'unfrozen', scheduler, writer=writer, start_epoch_offset=10)

    writer.close()

def train_phase(model, train_loader, test_loader, optimizer, criterion, device, epochs, best_acc, phase_name, scheduler=None, writer=None, start_epoch_offset=0):
    for epoch in range(1, epochs + 1):
        global_epoch = start_epoch_offset + epoch # Để hiển thị liên tục trên TensorBoard
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()
            
        if scheduler: scheduler.step()
        
        # Validation
        val_acc, val_loss = validate(model, test_loader, criterion, device)
        
        train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        lr = optimizer.param_groups[0]['lr']
        
        print(f"[{phase_name.upper()}] Ep {epoch}/{epochs} (Global {global_epoch}) | Loss: {train_loss:.4f}/{val_loss:.4f} | Acc: {train_acc:.2f}%/{val_acc:.2f}% | LR: {lr:.6f}")
        
        # Write to TensorBoard
        if writer:
            writer.add_scalar('Loss/Train', train_loss, global_epoch)
            writer.add_scalar('Loss/Val', val_loss, global_epoch)
            writer.add_scalar('Acc/Train', train_acc, global_epoch)
            writer.add_scalar('Acc/Val', val_acc, global_epoch)
            writer.add_scalar('LR', lr, global_epoch)

        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([phase_name, epoch, train_loss, val_loss, train_acc, val_acc, lr])

        if val_acc > best_acc:
            diff = val_acc - best_acc
            best_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  --> \033[92mNEW BEST: {best_acc:.2f}% (+{diff:.2f}%)\033[0m")

def validate(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # Simple TTA (Horizontal Flip)
            out1 = model(batch_x)
            out2 = model(torch.flip(batch_x, [3]))
            outputs = (out1 + out2) / 2.0
            
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()
            
    return 100 * correct / total, val_loss / len(loader)

if __name__ == "__main__":
    main()
