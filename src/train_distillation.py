import torch
import torch.nn as nn
import torch.nn.functional as F
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

# --- CẤU HÌNH DISTILLATION ---
BATCH_SIZE = 128
NUM_EPOCHS = 60           # Train ngắn vì Student đã thông minh sẵn (nếu load pretrained)
INPUT_SIZE = 48
NUM_CLASSES = 7
TEMP = 3.0                # Temperature: Làm mềm xác suất của Teacher
ALPHA = 0.5               # Cân bằng giữa học Teacher và học Label thật

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, '../trained_models/emotion_models/')
DATASET_PATH = os.path.join(SCRIPT_DIR, '../datasets')

# Teacher Paths (3 model best)
TEACHER_PATHS = [
    os.path.join(BASE_PATH, 'fer2013_resnet18_freeze_retrain_best.pth'),
    os.path.join(BASE_PATH, 'fer2013_resnet18_best_optimized.pth'),
    os.path.join(BASE_PATH, 'fer2013_resnet18_mixup_best.pth')
]

# Student Path (Khởi tạo từ model tốt nhất hiện tại để học cho nhanh)
STUDENT_INIT_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_freeze_retrain_best.pth')

# Output
TENSORBOARD_LOG_DIR = os.path.join(BASE_PATH, 'tensorboard_logs_distillation')
CSV_LOG_PATH = os.path.join(BASE_PATH, 'fer2013_resnet_distillation.csv')
BEST_MODEL_PATH = os.path.join(BASE_PATH, 'fer2013_resnet18_distilled_best.pth')

def load_model(path, device):
    model = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if not k.startswith('n_averaged'): new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval() # Teacher luôn ở chế độ Eval
    return model

def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # Loss 1: Hard Label (CrossEntropy chuẩn)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Loss 2: Soft Label (KL Divergence với Teacher)
    # Teacher logits cần được average trước nếu có nhiều teacher
    distill_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    
    return alpha * hard_loss + (1 - alpha) * distill_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | KNOWLEDGE DISTILLATION")

    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

    # 1. Load Teachers
    print("Loading Teachers...")
    teachers = []
    for path in TEACHER_PATHS:
        if os.path.exists(path):
            t = load_model(path, device)
            teachers.append(t)
            print(f"  -> Loaded Teacher: {os.path.basename(path)}")
    
    if not teachers:
        print("Error: No teachers found!")
        return

    # 2. Load Student
    print(f"Initializing Student from: {os.path.basename(STUDENT_INIT_PATH)}")
    student = resnet18_custom(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(STUDENT_INIT_PATH, map_location=device)
    # Load weights student giống như load teacher
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        if not k.startswith('n_averaged'): new_state_dict[name] = v
    student.load_state_dict(new_state_dict, strict=False)
    student.to(device)

    # 3. Datasets
    train_dataset = FERDataset(os.path.join(DATASET_PATH, 'train'), (INPUT_SIZE, INPUT_SIZE), mode='train')
    test_dataset = FERDataset(os.path.join(DATASET_PATH, 'test'), (INPUT_SIZE, INPUT_SIZE), mode='val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Optimizer
    # LR nhỏ thôi vì Student đã giỏi rồi, chỉ cần tinh chỉnh theo Teacher
    optimizer = SGD(student.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Log
    best_acc = 72.72 # Mốc cần vượt
    with open(CSV_LOG_PATH, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'loss', 'val_acc', 'lr'])

    print("Starting Distillation Training...")
    
    for epoch in range(1, NUM_EPOCHS + 1):
        student.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # 1. Get Teacher Outputs (Ensemble Logits)
            with torch.no_grad():
                teacher_logits_sum = torch.zeros_like(student(batch_x))
                for t_model in teachers:
                    teacher_logits_sum += t_model(batch_x)
                teacher_avg_logits = teacher_logits_sum / len(teachers)
            
            # 2. Get Student Output
            optimizer.zero_grad()
            student_logits = student(batch_x)
            
            # 3. Calc Loss
            loss = distillation_loss(student_logits, teacher_avg_logits, batch_y, TEMP, ALPHA)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        
        # Validation
        val_acc, _ = validate(student, test_loader, device)
        avg_loss = total_loss / len(train_loader)
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Distill Ep {epoch}/{NUM_EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}")
        
        writer.add_scalar('Distill/Loss', avg_loss, epoch)
        writer.add_scalar('Distill/Acc', val_acc, epoch)
        
        with open(CSV_LOG_PATH, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, val_acc, lr])

        if val_acc > best_acc:
            diff = val_acc - best_acc
            best_acc = val_acc
            torch.save(student.state_dict(), BEST_MODEL_PATH)
            print(f"  --> \033[92mNEW SINGLE MODEL RECORD: {best_acc:.2f}% (+{diff:.2f}%)\033[0m")

    writer.close()

def validate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            # TTA Flip
            out = (model(batch_x) + model(torch.flip(batch_x, [3]))) / 2.0
            _, pred = torch.max(out, 1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()
    return 100 * correct / total, 0

if __name__ == "__main__":
    main()
