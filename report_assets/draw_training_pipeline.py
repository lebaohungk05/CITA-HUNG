import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
import random
import os
import glob

def get_random_images(dataset_dir, count=2):
    try:
        pattern = os.path.join(dataset_dir, "**", "*.jpg")
        files = glob.glob(pattern, recursive=True)
        if len(files) < count:
            return None
        return random.sample(files, count)
    except Exception as e:
        print(f"Error finding images: {e}")
        return None

def draw_training_pipeline():
    # 1. Config & Data
    dataset_dir = r"face_classification/datasets/train"
    img_paths = get_random_images(dataset_dir, 2)
    
    if img_paths is None:
        print("Could not find enough images. Using placeholders.")
        img1 = np.zeros((48, 48, 3), dtype=np.uint8) + 50
        img2 = np.zeros((48, 48, 3), dtype=np.uint8) + 200
    else:
        img1 = cv2.imread(img_paths[0])
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.imread(img_paths[1])
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Simulate Mixup
    lam = 0.6
    mixup_img = cv2.addWeighted(img1, lam, img2, 1 - lam, 0)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    c_input = '#bdc3c7'
    c_backbone = '#2ecc71' # Green for active backbone
    c_se = '#9b59b6'       # Purple for SE Blocks
    c_head = '#3498db'     # Blue
    c_loss = '#e74c3c'     # Red
    c_sched = '#f39c12'    # Orange for Scheduler

    # Title
    ax.text(10, 9.5, "Super-Convergence Training Pipeline (SE-ResNet18)", fontsize=16, fontweight='bold', ha='center')

    # --- 1. MIXUP INPUT ---
    y_main = 5.0
    
    # Image 1 (Small)
    im_box1 = OffsetImage(img1, zoom=0.8)
    ab1 = AnnotationBbox(im_box1, (1.5, y_main + 1.5), frameon=True, bboxprops=dict(edgecolor='gray'))
    ax.add_artist(ab1)
    
    # Image 2 (Small)
    im_box2 = OffsetImage(img2, zoom=0.8)
    ab2 = AnnotationBbox(im_box2, (1.5, y_main - 1.5), frameon=True, bboxprops=dict(edgecolor='gray'))
    ax.add_artist(ab2)
    
    # Mixup Operation
    ax.text(2.5, y_main, "+", fontsize=20, ha='center', va='center')
    ax.text(2.5, y_main - 0.5, "Mixup\n(Beta Dist)", ha='center', va='top', fontsize=8)

    # Mixed Image
    im_box_mix = OffsetImage(mixup_img, zoom=1.2)
    ab_mix = AnnotationBbox(im_box_mix, (4.0, y_main), frameon=True, bboxprops=dict(edgecolor='black', linewidth=2))
    ax.add_artist(ab_mix)
    ax.text(4.0, y_main - 1.5, "Mixed Input\n(48x48)", ha='center', va='top', fontweight='bold')

    # Arrow to Model
    ax.arrow(5.2, y_main, 1.0, 0, head_width=0.2, fc='black', ec='black')

    # --- 2. SE-RESNET18 BACKBONE ---
    # Box
    rect_bb = patches.FancyBboxPatch((6.5, y_main - 1.5), 5.5, 3.0, boxstyle="round,pad=0.2", ec='black', fc=c_backbone, alpha=0.2)
    ax.add_patch(rect_bb)
    ax.text(9.25, y_main + 1.8, "SE-ResNet18 Custom", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Internal Layers (simplified)
    # Stem
    rect_stem = patches.Rectangle((6.8, y_main - 0.5), 0.8, 1.0, ec='black', fc='#95a5a6')
    ax.add_patch(rect_stem)
    ax.text(7.2, y_main, "Conv3x3\n(No Pool)", ha='center', va='center', fontsize=7, rotation=90, color='white')

    # Res Blocks + SE
    for i in range(4):
        x_start = 8.0 + i * 0.9
        # ResNet Block
        rect_res = patches.Rectangle((x_start, y_main - 1.0), 0.6, 2.0, ec='black', fc=c_backbone)
        ax.add_patch(rect_res)
        # SE Block indicator
        rect_se = patches.Rectangle((x_start + 0.1, y_main - 0.2), 0.4, 0.4, ec='black', fc=c_se)
        ax.add_patch(rect_se)
    
    ax.text(9.5, y_main - 1.2, "4 Stages (ResBlocks + SE)", ha='center', va='top', fontsize=9)
    ax.text(11.8, y_main, "SE Attention", color=c_se, fontsize=8, fontweight='bold', ha='right', va='bottom')

    # Arrow to Head
    ax.arrow(12.2, y_main, 0.8, 0, head_width=0.2, fc='black', ec='black')

    # --- 3. HEAD & LOSS ---
    # FC Head
    rect_head = patches.FancyBboxPatch((13.2, y_main - 0.8), 1.5, 1.6, boxstyle="round,pad=0.1", ec='black', fc=c_head)
    ax.add_patch(rect_head)
    ax.text(13.95, y_main, "FC Head\n(7 Classes)", ha='center', va='center', color='white', fontweight='bold')

    # Output Arrow
    ax.arrow(14.9, y_main, 0.8, 0, head_width=0.2, fc='black', ec='black')

    # Loss
    circle_loss = patches.Circle((16.5, y_main), 0.6, fc=c_loss, ec='black')
    ax.add_patch(circle_loss)
    ax.text(16.5, y_main, "Loss", ha='center', va='center', color='white', fontweight='bold')
    ax.text(16.5, y_main - 0.8, "CrossEntropy\n+ Label Smooth", ha='center', va='top', fontsize=9)

    # --- 4. SCHEDULER & OPTIMIZER ---
    # Optimizer Box
    rect_opt = patches.FancyBboxPatch((14.0, 1.5), 4.0, 1.5, boxstyle="round,pad=0.1", ec='black', fc='#ecf0f1')
    ax.add_patch(rect_opt)
    
    ax.text(16.0, 2.7, "Optimizer: AdamW", ha='center', fontweight='bold')
    
    # Draw OneCycle Curve
    x_curve = np.linspace(14.5, 17.5, 50)
    # Bell curve approx
    y_curve = 1.8 + 0.8 * np.exp(-0.5 * ((x_curve - 16.0) / 0.5)**2) 
    ax.plot(x_curve, y_curve, color=c_sched, linewidth=2)
    ax.text(16.0, 1.6, "OneCycleLR Schedule", ha='center', fontsize=8, color=c_sched)

    # Connect Optimizer to Update
    ax.annotate("", xy=(12.0, 2.25), xytext=(14.0, 2.25), arrowprops=dict(arrowstyle="->", ls="dashed", color="gray"))
    ax.text(13.0, 2.4, "Update Gradients", ha='center', fontsize=9, color='gray')

    # Legend
    legend_elements = [
        patches.Patch(facecolor=c_backbone, edgecolor='black', label='Trainable Backbone'),
        patches.Patch(facecolor=c_se, edgecolor='black', label='SE Attention Block'),
        patches.Patch(facecolor=c_sched, edgecolor='none', label='Learning Rate Curve'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

    plt.tight_layout()
    plt.savefig('project_pipeline.png', dpi=300)
    print(f"Saved corrected visualization to project_pipeline.png")

if __name__ == "__main__":
    draw_training_pipeline()
