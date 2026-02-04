import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2
import numpy as np
import os
import glob

def draw_3d_box(ax, x, y, w, h, d, fc, ec='#2c3e50', alpha=1.0, zorder=10):
    ax.add_patch(patches.Rectangle((x, y), w, h, fc=fc, ec=ec, alpha=alpha, zorder=zorder))
    pts_top = np.array([[x, y+h], [x+d, y+h+d*0.6], [x+w+d, y+h+d*0.6], [x+w, y+h]])
    ax.add_patch(patches.Polygon(pts_top, fc=fc, ec=ec, alpha=alpha*0.8, zorder=zorder-1))
    pts_side = np.array([[x+w, y], [x+w+d, y+d*0.6], [x+w+d, y+h+d*0.6], [x+w, y+h]])
    ax.add_patch(patches.Polygon(pts_side, fc=fc, ec=ec, alpha=alpha*0.6, zorder=zorder-1))

def draw_thick_arrow(ax, x, y, length, width=0.6, color='#7f8c8d'):
    ax.arrow(x, y, length, 0, width=width, head_width=width*2.5, head_length=width*2, 
             fc=color, ec=color, length_includes_head=True, zorder=5)

def get_specific_mixup_data():
    happy_path = r"face_classification/datasets/test/happy/PrivateTest_10077120.jpg"
    angry_path = r"face_classification/datasets/test/angry/PrivateTest_10131363.jpg"
    
    if os.path.exists(happy_path) and os.path.exists(angry_path):
        img_happy = cv2.imread(happy_path)
        img_angry = cv2.imread(angry_path)
    else:
        img_happy = np.zeros((48,48,3), dtype=np.uint8) + 200
        img_angry = np.zeros((48,48,3), dtype=np.uint8) + 50

    img_happy = cv2.resize(img_happy, (48, 48))
    img_angry = cv2.resize(img_angry, (48, 48))
    img_mix = cv2.addWeighted(img_happy, 0.6, img_angry, 0.4, 0)

    img_happy = cv2.cvtColor(img_happy, cv2.COLOR_BGR2RGB)
    img_angry = cv2.cvtColor(img_angry, cv2.COLOR_BGR2RGB)
    img_mix = cv2.cvtColor(img_mix, cv2.COLOR_BGR2RGB)
    
    return img_happy, img_angry, img_mix

def draw_pipeline_architecture():
    # --- CANVAS SETUP ---
    fig, ax = plt.subplots(figsize=(115, 32)) 
    ax.set_xlim(-20, 103)  
    ax.set_ylim(-5, 28)
    ax.axis('off')

    c_stage1, c_stage2, c_stage3, c_stage4 = '#1abc9c', '#3498db', '#9b59b6', '#e91e63'
    c_gap, c_vec, c_node_green = '#f1c40f', '#e67e22', '#2ecc71'

    # --- DATA PREP ---
    img_happy, img_angry, img_mix = get_specific_mixup_data()
    y_center = 14.0

    # =========================================================================
    # PART 1: MIXUP VISUALIZATION (LEFT)
    # =========================================================================
    x_origin = -14.0
    y_origin_top = y_center + 6.0
    y_origin_bot = y_center - 6.0

    ax.add_artist(AnnotationBbox(OffsetImage(img_happy, zoom=6.5), (x_origin, y_origin_top), frameon=True, bboxprops=dict(edgecolor='#2ecc71', lw=4)))
    ax.text(x_origin, y_origin_top + 5.5, "Image A (Happy)", ha='center', fontsize=60, fontweight='bold')

    ax.add_artist(AnnotationBbox(OffsetImage(img_angry, zoom=6.5), (x_origin, y_origin_bot), frameon=True, bboxprops=dict(edgecolor='#e74c3c', lw=4)))
    ax.text(x_origin, y_origin_bot - 6.5, "Image B (Angry)", ha='center', fontsize=60, fontweight='bold')

    ax.text(x_origin + 5.0, y_center, "+", ha='center', va='center', fontsize=120, fontweight='bold', color='#7f8c8d')
    ax.annotate("", xy=(-2.0, y_center), xytext=(x_origin + 4.0, y_origin_top), arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=5, mutation_scale=40))
    ax.annotate("", xy=(-2.0, y_center), xytext=(x_origin + 4.0, y_origin_bot), arrowprops=dict(arrowstyle="->", color='#7f8c8d', lw=5, mutation_scale=40))

    input_x = 2.0 
    ax.add_artist(AnnotationBbox(OffsetImage(img_mix, zoom=10.0), (input_x, y_center), frameon=False))
    ax.text(input_x, y_center - 7.5, "Mixup Input\n(48x48)", ha='center', va='top', fontsize=95, fontweight='bold')
    draw_thick_arrow(ax, input_x + 5.0, y_center, 5.0)

    # =========================================================================
    # PART 2: BACKBONE (MIDDLE)
    # =========================================================================
    pre_x = 15.0  
    ax.add_artist(AnnotationBbox(OffsetImage(img_mix, zoom=6.5), (pre_x, y_center), frameon=False))
    ax.text(pre_x, y_center - 6.5, "Preprocessing", ha='center', va='top', fontsize=95)
    draw_thick_arrow(ax, pre_x + 4.0, y_center, 5.0)

    bw, bh = 5.0, 7.0
    s1_x = 24.0
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
    ax.text(s2_x + bw, y_center + 11.0, "SE-ResNet18 Backbone", ha='center', fontsize=105, fontweight='bold')

    stage4_right_x = s4_x + bw + 3.5 
    vec_x = stage4_right_x + 6.0 
    vec_h, vec_w = 2.5, 9.0
    gap_poly = np.array([[stage4_right_x, y_center + 5.5], [stage4_right_x, y_center - 5.5], [vec_x, y_center - vec_h/2], [vec_x, y_center + vec_h/2]])
    ax.add_patch(patches.Polygon(gap_poly, fc=c_gap, alpha=0.4, ec=None))
    ax.text((stage4_right_x + vec_x)/2, y_center - 7.5, "Global Avg Pooling", ha='center', fontsize=90)
    draw_3d_box(ax, vec_x, y_center - vec_h/2, vec_w, vec_h, 0.8, c_vec)
    ax.text(vec_x + vec_w/2, y_center - 4.5, "Vector [512]", ha='center', fontsize=90)

    l1_x = vec_x + vec_w + 5.0 
    l1_y_range = np.linspace(y_center - 7.0, y_center + 7.0, 8)
    l2_x = l1_x + 5.0  
    l2_y_range = np.linspace(y_center - 4.0, y_center + 4.0, 4)
    out_head_x = l2_x + 5.0  
    
    for y1 in l1_y_range:
        for y2 in l2_y_range: ax.plot([l1_x, l2_x], [y1, y2], color='gray', lw=2.0, alpha=0.4, zorder=5)
    for y1 in l1_y_range:
        ax.add_patch(patches.Circle((l1_x, y1), 0.55, fc=c_node_green, ec=None, zorder=20))
        ax.plot([vec_x + vec_w + 0.8, l1_x], [y_center, y1], color='gray', lw=2.0, alpha=0.4, zorder=5)

    ax.text((l1_x + l2_x)/2, y_center - 11.0, "Fully Connected", ha='center', fontsize=95, color='gray', fontweight='bold')

    # =========================================================================
    # PART 3: OUTPUT VECTOR & LABELS (RIGHT) - MODIFIED V3
    # =========================================================================
    
    start_vec_x = out_head_x + 3.0
    connect_x = start_vec_x - 0.5
    for y2 in l2_y_range:
        ax.add_patch(patches.Circle((l2_x, y2), 0.55, fc=c_node_green, ec=None, zorder=20))
        ax.plot([l2_x, connect_x], [y2, y_center], color='gray', lw=2.5, alpha=0.6, zorder=5)

    ax.text(start_vec_x + 7.0, y_center + 12.0, "Class Reliability (F1-Score)", ha='center', fontsize=85, fontweight='bold')

    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    confidences = [0.66, 0.77, 0.58, 0.90, 0.69, 0.60, 0.83] 
    
    start_y = y_center + 8.0 
    spacing_y = 3.3 
    
    # Increase vector width to fit text
    vec_width = 5.0 # Wider
    
    for i, (emo, conf) in enumerate(zip(emotions, confidences)):
        y_pos = start_y - i * spacing_y
        
        fill_color = plt.cm.Blues(conf)
        if emo == 'happy': fill_color = '#2ecc71'

        cell_h = 2.8
        rect = patches.Rectangle((start_vec_x, y_pos - cell_h/2), vec_width, cell_h, 
                                 fc=fill_color, ec='black', lw=2.5)
        ax.add_patch(rect)
        
        # --- KEY CHANGE: TEXT INSIDE RECT ---
        text_color_inside = 'white' if conf > 0.6 else 'black'
        ax.text(start_vec_x + vec_width/2, y_pos, f"{conf*100:.0f}%", 
                ha='center', va='center', fontsize=65, fontweight='bold', color=text_color_inside)
        
        # Draw Arrow
        arrow_start = start_vec_x + vec_width
        ax.annotate("", xy=(arrow_start + 2.5, y_pos), xytext=(arrow_start, y_pos),
                    arrowprops=dict(arrowstyle="-", color='black', lw=2.0))
        
        # Label Text (ONLY NAME)
        fontweight, color = 'normal', 'black'
        if emo == 'happy':
            fontweight, color = 'bold', '#27ae60'
            ax.text(arrow_start + 2.5, y_pos, "◄", ha='left', va='center', color=color, fontsize=90)

        # Capitalize and remove % from here
        ax.text(arrow_start + 4.0, y_pos, emo.capitalize(), ha='left', va='center', fontsize=65, fontweight=fontweight, color=color)

    ax.text(start_vec_x + vec_width/2, start_y - 7 * spacing_y + 1.0, "Output\nVector", ha='center', va='top', fontsize=60, color='#546e7a')

    plt.tight_layout()
    plt.savefig('pipeline_of_the_proposed_SE-ResNet18_backbone_v2.png', dpi=90, bbox_inches='tight')
    print("Saved to pipeline_of_the_proposed_SE-ResNet18_backbone_v2.png")

if __name__ == "__main__":
    draw_pipeline_architecture()