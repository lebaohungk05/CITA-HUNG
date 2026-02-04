import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

# Cấu hình đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, 'trained_models', 'emotion_models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'report_images')

LOG_1 = os.path.join(LOG_DIR, 'fer2013_resnet_training_log_mixup_optimized.csv')
LOG_2 = os.path.join(LOG_DIR, 'fer2013_resnet_freeze_retrain.csv')

def smooth(scalars, weight=0.6):
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def main():
    if not os.path.exists(LOG_1) or not os.path.exists(LOG_2):
        print("Error: Log files not found.")
        return

    # 1. Đọc dữ liệu
    df1 = pd.read_csv(LOG_1)
    df2 = pd.read_csv(LOG_2)

    print(f"Log 1 (Optimized): {len(df1)} epochs")
    print(f"Log 2 (Finetune): {len(df2)} epochs")

    # 2. Xử lý nối dữ liệu
    # df1 chạy từ epoch 1 -> 265
    # df2 chạy từ epoch 1 -> 20 (hoặc 30)
    # Ta sẽ offset epoch của df2 để nối tiếp vào df1
    last_epoch_1 = df1['epoch'].iloc[-1]
    df2['epoch'] = df2['epoch'] + last_epoch_1

    # Gộp DataFrame
    # Chỉ lấy các cột chung quan trọng
    cols = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
    df_combined = pd.concat([df1[cols], df2[cols]], ignore_index=True)

    print(f"Combined: {len(df_combined)} epochs")

    # 3. Vẽ đồ thị (Style TensorBoard)
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 300
    
    # --- CHART 1: ACCURACY ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train Acc
    ax1.plot(df_combined['epoch'], df_combined['train_acc'], color='#ff7f0e', alpha=0.3, linewidth=1)
    ax1.plot(df_combined['epoch'], smooth(df_combined['train_acc']), color='#ff7f0e', linewidth=2)
    ax1.set_title('Training Accuracy', fontsize=14, fontweight='bold', color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.grid(True, linestyle='-', alpha=0.2)
    
    # Val Acc
    ax2.plot(df_combined['epoch'], df_combined['val_acc'], color='#1f77b4', alpha=0.3, linewidth=1)
    ax2.plot(df_combined['epoch'], smooth(df_combined['val_acc']), color='#1f77b4', linewidth=2)
    
    # Đánh dấu điểm chuyển giao (Fine-tuning start)
    ax2.axvline(x=last_epoch_1, color='gray', linestyle='--', alpha=0.5, label='Start Fine-tuning')
    
    # --- MARKING POINTS ---
    # 1. Best của Phase 1 (Optimized)
    max_acc_1 = df1['val_acc'].max()
    best_epoch_1 = df1.loc[df1['val_acc'].idxmax(), 'epoch']
    
    ax2.scatter(best_epoch_1, max_acc_1, color='orange', s=50, zorder=5)
    ax2.annotate(f'{max_acc_1:.2f}%', (best_epoch_1, max_acc_1 + 0.5), color='orange', fontweight='bold', ha='right')
    ax2.vlines(x=best_epoch_1, ymin=ax2.get_ylim()[0], ymax=max_acc_1, color='orange', linestyle=':', alpha=0.6)
    
    # Label Ep 1 dưới trục hoành
    ax2.text(best_epoch_1, -0.08, f'Ep{int(best_epoch_1)}', color='orange', fontweight='bold', 
             ha='center', transform=ax2.get_xaxis_transform())

    # 2. Best Global (Fine-tuned)
    max_acc_2 = df_combined['val_acc'].max()
    best_epoch_2 = df_combined.loc[df_combined['val_acc'].idxmax(), 'epoch']
    
    ax2.scatter(best_epoch_2, max_acc_2, color='red', s=50, zorder=5)
    ax2.annotate(f'{max_acc_2:.2f}%', (best_epoch_2, max_acc_2 + 0.5), color='red', fontweight='bold', ha='left')
    ax2.vlines(x=best_epoch_2, ymin=ax2.get_ylim()[0], ymax=max_acc_2, color='red', linestyle=':', alpha=0.6)
    
    # Label Ep 2 dưới trục hoành
    ax2.text(best_epoch_2, -0.08, f'Ep{int(best_epoch_2)}', color='red', fontweight='bold', 
             ha='center', transform=ax2.get_xaxis_transform())

    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold', color='#1f77b4')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, linestyle='-', alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_accuracy_chart.png'), bbox_inches='tight')
    print("Saved combined_accuracy_chart.png")
    plt.close()

    # --- CHART 2: LOSS ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Train Loss
    ax1.plot(df_combined['epoch'], df_combined['train_loss'], color='#ff7f0e', alpha=0.3, linewidth=1)
    ax1.plot(df_combined['epoch'], smooth(df_combined['train_loss']), color='#ff7f0e', linewidth=2)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold', color='#ff7f0e')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, linestyle='-', alpha=0.2)
    
    # Val Loss
    ax2.plot(df_combined['epoch'], df_combined['val_loss'], color='#1f77b4', alpha=0.3, linewidth=1)
    ax2.plot(df_combined['epoch'], smooth(df_combined['val_loss']), color='#1f77b4', linewidth=2)
    
    # Đánh dấu chuyển giao
    ax2.axvline(x=last_epoch_1, color='gray', linestyle='--', alpha=0.5, label='Start Fine-tuning')

    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold', color='#1f77b4')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, linestyle='-', alpha=0.2)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_loss_chart.png'), bbox_inches='tight')
    print("Saved combined_loss_chart.png")
    plt.close()

if __name__ == "__main__":
    main()
