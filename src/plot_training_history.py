import pandas as pd
import matplotlib
matplotlib.use('Agg') # Set non-GUI backend
import matplotlib.pyplot as plt
import os
import numpy as np

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(BASE_DIR, 'trained_models', 'emotion_models', 'fer2013_resnet_training_log_mixup.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'report_images')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_training_history(log_path):
    print(f"Reading log file from: {log_path}")
    if not os.path.exists(log_path):
        print(f"Error: File not found at {log_path}")
        return

    # Đọc dữ liệu
    try:
        df = pd.read_csv(log_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # Thiết lập style giống TensorBoard (nền trắng, grid nhẹ)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.style.use('default') 
    
    # Hàm làm mượt (Smoothing) giống TensorBoard
    def smooth(scalars, weight=0.6):  # Weight 0.6 là mức mặc định của TensorBoard
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # --- 1. BIỂU ĐỒ ACCURACY (2 Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Train Accuracy
    ax1.plot(df['epoch'], df['train_acc'], color='#ff7f0e', alpha=0.3, linewidth=1)
    ax1.plot(df['epoch'], smooth(df['train_acc']), color='#ff7f0e', linewidth=2)
    ax1.set_title('Training Accuracy', fontsize=14, fontweight='bold', color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.grid(True, linestyle='-', alpha=0.2)

    # Subplot 2: Validation Accuracy
    ax2.plot(df['epoch'], df['val_acc'], color='#1f77b4', alpha=0.3, linewidth=1)
    ax2.plot(df['epoch'], smooth(df['val_acc']), color='#1f77b4', linewidth=2)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold', color='#1f77b4')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='-', alpha=0.2)
    
    plt.tight_layout()
    acc_path = os.path.join(OUTPUT_DIR, 'final_accuracy_chart.png')
    plt.savefig(acc_path, bbox_inches='tight')
    print(f"Saved Accuracy chart to {acc_path}")
    plt.close()

    # --- 2. BIỂU ĐỒ LOSS (2 Subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Train Loss
    ax1.plot(df['epoch'], df['train_loss'], color='#ff7f0e', alpha=0.3, linewidth=1)
    ax1.plot(df['epoch'], smooth(df['train_loss']), color='#ff7f0e', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold', color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='-', alpha=0.2)
    
    # Subplot 2: Validation Loss
    ax2.plot(df['epoch'], df['val_loss'], color='#1f77b4', alpha=0.3, linewidth=1)
    ax2.plot(df['epoch'], smooth(df['val_loss']), color='#1f77b4', linewidth=2)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold', color='#1f77b4')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='-', alpha=0.2)

    plt.tight_layout()
    loss_path = os.path.join(OUTPUT_DIR, 'final_loss_chart.png')
    plt.savefig(loss_path, bbox_inches='tight')
    print(f"Saved Loss chart to {loss_path}")
    plt.close()

if __name__ == "__main__":
    plot_training_history(LOG_FILE)
