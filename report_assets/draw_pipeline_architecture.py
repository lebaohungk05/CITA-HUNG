import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
import os
import glob
import random
import sys
import torch
import torch.nn.functional as F

# Add src to path to import models
sys.path.append(os.path.join(os.getcwd(), 'face_classification', 'src'))
from models.resnet18_custom import resnet18_custom

def load_model(device):
    model_path = r"face_classification/trained_models/emotion_models/fer2013_resnet18_mixup_best.pth"
    model = resnet18_custom(num_classes=7, pretrained=False)
    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
            else: model.load_state_dict(checkpoint)
        except Exception: pass
    model.to(device); model.eval()
    return model

def get_mixup_image(dataset_dir):
    try:
        pattern = os.path.join(dataset_dir, "**", "*.jpg")
        files = glob.glob(pattern, recursive=True)
        if len(files) < 2: return None
        p1, p2 = random.sample(files, 2)
        i1 = cv2.imread(p1); i2 = cv2.imread(p2)
        i1 = cv2.resize(i1, (48, 48)); i2 = cv2.resize(i2, (48, 48))
        i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB); i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(i1, 0.6, i2, 0.4, 0)
    except Exception: return None

def get_emotion_representative_images(dataset_root):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    images = {}
    for emo in emotions:
        path = os.path.join(dataset_root, 'test', emo, '*.jpg')
        files = glob.glob(path)
        if files:
            img = cv2.imread(random.choice(files))
            if img is not None: images[emo] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if emo not in images: images[emo] = np.zeros((48, 48, 3), dtype=np.uint8)
    return images, emotions

def draw_3d_box(ax, x, y, w, h, d, fc, ec='#2c3e50', alpha=1.0, zorder=10):
    ax.add_patch(patches.Rectangle((x, y), w, h, fc=fc, ec=ec, alpha=alpha, zorder=zorder))
    pts_top = np.array([[x, y+h], [x+d, y+h+d*0.6], [x+w+d, y+h+d*0.6], [x+w, y+h]])
    ax.add_patch(patches.Polygon(pts_top, fc=fc, ec=ec, alpha=alpha*0.8, zorder=zorder-1))
    pts_side = np.array([[x+w, y], [x+w+d, y+d*0.6], [x+w+d, y+h+d*0.6], [x+w, y+h]])
    ax.add_patch(patches.Polygon(pts_side, fc=fc, ec=ec, alpha=alpha*0.6, zorder=zorder-1))

def draw_thick_arrow(ax, x, y, length, width=0.6):
    ax.arrow(x, y, length, 0, width=width, head_width=width*2.5, head_length=width*2, 
             fc='#7f8c8d', ec='#7f8c8d', length_includes_head=True, zorder=5)

def draw_pipeline_architecture():
    # Adjusted layout for paper - more compact, shifted left, padded right
    fig, ax = plt.subplots(figsize=(105, 32))  # Adjusted width for emotion labels
    ax.set_xlim(-2, 103)  # Extended right side for emotion images + padding
    ax.set_ylim(-5, 28)
    ax.axis('off')

    c_stage1, c_stage2, c_stage3, c_stage4 = '#1abc9c', '#3498db', '#9b59b6', '#e91e63'
    c_gap, c_vec, c_node_green = '#f1c40f', '#e67e22', '#2ecc71'

    dataset_dir = r"face_classification/datasets/train"
    mixup_img = get_mixup_image(dataset_dir)
    if mixup_img is None: mixup_img = np.zeros((48, 48, 3), dtype=np.uint8) + 128
    
    y_center = 14.0 
    
    # --- 1. INPUT --- (shifted left)
    input_x = 2.0  # Moved from 7.0 to 2.0
    ax.add_artist(AnnotationBbox(OffsetImage(mixup_img, zoom=10.0), (input_x, y_center), frameon=False))
    ax.text(input_x, y_center - 7.5, "Mixup Input\n(48x48)", ha='center', va='top', fontsize=95, fontweight='bold')
    draw_thick_arrow(ax, input_x + 5.0, y_center, 5.0)

    # --- 2. PREPROCESSING --- (shifted left)
    pre_x = 15.0  # Moved from 20.0 to 15.0
    ax.add_artist(AnnotationBbox(OffsetImage(mixup_img, zoom=6.5), (pre_x, y_center), frameon=False))
    ax.text(pre_x, y_center - 6.5, "Preprocessing", ha='center', va='top', fontsize=95)
    draw_thick_arrow(ax, pre_x + 4.0, y_center, 5.0)

    # --- 3. BACKBONE --- (shifted left)
    bw, bh = 5.0, 7.0
    s1_x = 24.0  # Moved from 29.0 to 24.0
    draw_3d_box(ax, s1_x, y_center-bh/2, bw, bh, 1.8, c_stage1) 
    ax.text(s1_x + bw/2, y_center - bh/2 - 4.0, "48x48x64", ha='center', va='top', fontsize=95, fontweight='bold')
    draw_thick_arrow(ax, s1_x + bw + 1.2, y_center, 4.0)
    
    s2_x = s1_x + bw + 5.5
    bh2 = bh + 1.5
    draw_3d_box(ax, s2_x, y_center-bh2/2, bw, bh2, 1.8, c_stage2)
    ax.text(s2_x + bw/2, y_center - bh2/2 - 4.0, "24x24x128", ha='center', va='top', fontsize=95, fontweight='bold')
    draw_thick_arrow(ax, s2_x + bw + 1.2, y_center, 4.0)
    
    s3_x = s2_x + bw + 5.5
    bh3 = bh2 + 1.5
    draw_3d_box(ax, s3_x, y_center-bh3/2, bw, bh3, 1.8, c_stage3)
    ax.text(s3_x + bw/2, y_center - bh3/2 - 4.0, "12x12x256", ha='center', va='top', fontsize=95, fontweight='bold')
    draw_thick_arrow(ax, s3_x + bw + 1.2, y_center, 4.0)
    
    s4_x = s3_x + bw + 5.5
    bh4 = bh3 + 1.5
    draw_3d_box(ax, s4_x, y_center-bh4/2, bw, bh4, 1.8, c_stage4)
    ax.text(s4_x + bw/2, y_center - bh4/2 - 4.0, "6x6x512", ha='center', va='top', fontsize=95, fontweight='bold')
    ax.text(s2_x + bw, y_center + 11.0, "SE-ResNet18 Backbone (Custom)", ha='center', fontsize=105, fontweight='bold')

    # --- 4. GAP ---
    stage4_right_x = s4_x + bw + 3.5 
    vec_x = stage4_right_x + 6.0 
    vec_h, vec_w = 2.5, 9.0
    gap_poly = np.array([
        [stage4_right_x, y_center + 5.5], [stage4_right_x, y_center - 5.5], 
        [vec_x, y_center - vec_h/2], [vec_x, y_center + vec_h/2]       
    ])
    ax.add_patch(patches.Polygon(gap_poly, fc=c_gap, alpha=0.4, ec=None))
    ax.text((stage4_right_x + vec_x)/2, y_center - 7.5, "Global Avg Pooling", ha='center', fontsize=90) # Increased
    draw_3d_box(ax, vec_x, y_center - vec_h/2, vec_w, vec_h, 0.8, c_vec)
    ax.text(vec_x + vec_w/2, y_center - 4.5, "Vector [512]", ha='center', fontsize=90) # Increased

    # --- 5. FC ---
    l1_x = vec_x + vec_w + 5.0 
    l1_y_range = np.linspace(y_center - 7.0, y_center + 7.0, 8)
    l2_x = l1_x + 5.0  # Reduced from 6.0 to shorten FC layer
    l2_y_range = np.linspace(y_center - 4.0, y_center + 4.0, 4)
    out_head_x = l2_x + 5.0  # Reduced from 8.0 to shorten FC layer
    
    for y1 in l1_y_range:
        for y2 in l2_y_range: ax.plot([l1_x, l2_x], [y1, y2], color='gray', lw=2.0, alpha=0.4, zorder=5)
    for y1 in l1_y_range:
        ax.add_patch(patches.Circle((l1_x, y1), 0.55, fc=c_node_green, ec=None, zorder=20))
        ax.plot([vec_x + vec_w + 0.8, l1_x], [y_center, y1], color='gray', lw=2.0, alpha=0.4, zorder=5)
    
    # Calculate connect point for images - move images further right to avoid overlap
    start_img_x = out_head_x + 5.0  # Increased from 0.5 to 5.0 to give more space
    # Connect lines to center point (not to each emotion)
    connect_x = start_img_x - 2.0
    
    # --- 6. OUTPUT ---
    dataset_root = "face_classification/datasets"
    emotion_images, emotion_labels = get_emotion_representative_images(dataset_root)
    confidences = {'angry': 0.66, 'disgust': 0.77, 'fear': 0.58, 'happy': 0.90, 'neutral': 0.69, 'sad': 0.60, 'surprise': 0.83}
    
    start_img_y = y_center + 8.0 
    img_spacing_y = 3.3 
    
    # Draw FC layer 2 neurons with lines to CENTER point (original behavior)
    for y2 in l2_y_range:
        ax.add_patch(patches.Circle((l2_x, y2), 0.55, fc=c_node_green, ec=None, zorder=20))
        ax.plot([l2_x, connect_x], [y2, y_center], color='gray', lw=2.5, alpha=0.6, zorder=5)

    ax.text((l1_x + l2_x)/2, y_center - 11.0, "Fully Connected", ha='center', fontsize=95, color='gray', fontweight='bold')
    
    # Title for emotion output section
    ax.text(start_img_x + 1.5, start_img_y + 3.5, "Class Reliability (F1-Score)", ha='center', fontsize=85, fontweight='bold')
    
    for i, emo in enumerate(emotion_labels):
        x_pos, y_pos = start_img_x, start_img_y - i * img_spacing_y
        img = emotion_images[emo]
        if img is not None:
            ab = AnnotationBbox(OffsetImage(img, zoom=3.5), (x_pos, y_pos), frameon=True, 
                              bboxprops=dict(edgecolor='gray', alpha=0.8, linewidth=4), pad=0.1, zorder=30)
            ax.add_artist(ab)
        
        conf = confidences[emo]
        label_text, fontweight, color = f"{emo.capitalize()} : {conf*100:.0f}%", 'normal', 'black'
        if emo == 'happy':
            fontweight, color = 'bold', '#27ae60'
            ax.text(x_pos + 2.5, y_pos, "◄", ha='left', va='center', color=color, fontsize=90)  # Reduced gap
        ax.text(x_pos + 3.5, y_pos, label_text, ha='left', va='center', fontsize=65, fontweight=fontweight, color=color)  # Reduced gap

    plt.tight_layout()
    plt.savefig('pipeline_of_the_proposed_SE-ResNet18_backbone.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    draw_pipeline_architecture()
