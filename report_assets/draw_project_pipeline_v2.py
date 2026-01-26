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

def get_random_image(dataset_dir, emotion='happy'):
    try:
        pattern = os.path.join(dataset_dir, "test", emotion, "*.jpg")
        files = glob.glob(pattern)
        if not files: return None
        path = random.choice(files)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None

def create_dashboard_mockup():
    """Create a synthetic image representing the Teacher Dashboard"""
    # Create a white canvas
    h, w = 300, 500
    img = np.ones((h, w, 3), dtype=np.uint8) * 245 # Light gray bg
    
    # Header
    cv2.rectangle(img, (0, 0), (w, 40), (52, 73, 94), -1) # Dark Blue Header
    cv2.putText(img, "Student Engagement Dashboard", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Chart Area
    cv2.rectangle(img, (20, 60), (320, 200), (255, 255, 255), -1)
    # Draw a line chart line
    points = []
    for x in range(20, 320, 30):
        y = random.randint(80, 180)
        points.append((x, y))
    
    for i in range(len(points)-1):
        cv2.line(img, points[i], points[i+1], (46, 204, 113), 2) # Green line
    
    # Stats Area (Right side)
    cv2.rectangle(img, (340, 60), (480, 120), (255, 255, 255), -1)
    cv2.putText(img, "Engagement", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(img, "85%", (360, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (46, 204, 113), 2)
    
    cv2.rectangle(img, (340, 140), (480, 200), (255, 255, 255), -1)
    cv2.putText(img, "Current State", (350, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.putText(img, "Happy", (360, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (231, 76, 60), 2)
    
    # Logs Area
    cv2.rectangle(img, (20, 220), (480, 280), (30, 30, 30), -1)
    cv2.putText(img, "> [10:05:01] Student 1: Happy (0.98)", (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(img, "> [10:05:02] Student 1: Happy (0.99)", (30, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    return img

def draw_project_pipeline_v2():
    # Setup canvas
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Data Loading
    dataset_dir = r"face_classification/datasets"
    raw_img = get_random_image(dataset_dir)
    if raw_img is None: 
        raw_img = np.zeros((100, 100, 3), dtype=np.uint8) + 200 # Fallback
    
    # --- 1. INPUT STREAM (Image) ---
    x_step = 3.5
    y_center = 4.0
    
    # Draw "Webcam" frame
    ax.text(1.5, 6.5, "1. Input Stream", ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    im_box1 = OffsetImage(raw_img, zoom=0.6) # Adjust zoom
    ab1 = AnnotationBbox(im_box1, (1.5, y_center), frameon=True, bboxprops=dict(edgecolor='#3498db', linewidth=3))
    ax.add_artist(ab1)
    
    # Arrow
    ax.annotate("", xy=(3.2, y_center), xytext=(2.5, y_center), 
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))

    # --- 2. DETECTION (Image with Box) ---
    ax.text(5.0, 6.5, "2. Face Detection\n(YuNet)", ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Create detection image (Draw box on raw image)
    det_img = raw_img.copy()
    h, w, _ = det_img.shape
    cv2.rectangle(det_img, (int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8)), (255, 165, 0), 3) # Orange box
    
    im_box2 = OffsetImage(det_img, zoom=0.6)
    ab2 = AnnotationBbox(im_box2, (5.0, y_center), frameon=True, bboxprops=dict(edgecolor='#e67e22', linewidth=3))
    ax.add_artist(ab2)

    # Arrow
    ax.annotate("", xy=(6.8, y_center), xytext=(6.0, y_center), 
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))

    # --- 3. PREPROCESSING (Gray Crop) ---
    ax.text(8.5, 6.5, "3. Preprocessing\n(48x48 Grayscale)", ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Create preproc image
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
    gray_img = cv2.resize(gray_img, (48, 48))
    # Make it 3 channel for display
    gray_disp = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    
    im_box3 = OffsetImage(gray_disp, zoom=1.5) # Zoom in because it's small
    ab3 = AnnotationBbox(im_box3, (8.5, y_center), frameon=True, bboxprops=dict(edgecolor='#95a5a6', linewidth=2, linestyle='--'))
    ax.add_artist(ab3)

    # Arrow
    ax.annotate("", xy=(10.2, y_center), xytext=(9.5, y_center), 
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))

    # --- 4. MODEL (ResNet Block) ---
    ax.text(12.0, 6.5, "4. Recognition Model\n(SE-ResNet18 Mixup)", ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    
    # Draw a stylish "Black Box"
    rect = patches.FancyBboxPatch((11.0, 3.0), 2.0, 2.0, boxstyle="round,pad=0.1", fc='#8e44ad', ec='#2c3e50', lw=2)
    ax.add_patch(rect)
    ax.text(12.0, 4.0, "CNN\nLayers", ha='center', va='center', color='white', fontweight='bold')
    
    # Output Probability Bars
    # Small bar chart next to model
    labels = ['Ang', 'Hap', 'Neu']
    vals = [0.1, 0.8, 0.1]
    cols = ['#e74c3c', '#2ecc71', '#95a5a6']
    
    for i, (val, col) in enumerate(zip(vals, cols)):
        bar_x = 13.5
        bar_y = 3.2 + i*0.6
        rect_bar = patches.Rectangle((bar_x, bar_y), val*1.5, 0.4, fc=col)
        ax.add_patch(rect_bar)
        ax.text(bar_x - 0.4, bar_y + 0.1, labels[i], fontsize=8)

    # Arrow
    ax.annotate("", xy=(15.8, y_center), xytext=(15.0, y_center), 
                arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=2))

    # --- 5. APP / DASHBOARD ---
    ax.text(17.5, 6.5, "5. Teacher Dashboard\n(Real-time Analytics)", ha='center', fontsize=12, fontweight='bold', color='#2c3e50')
    
    dashboard_img = create_dashboard_mockup()
    im_box5 = OffsetImage(dashboard_img, zoom=0.5)
    ab5 = AnnotationBbox(im_box5, (17.5, y_center), frameon=True, bboxprops=dict(edgecolor='#27ae60', linewidth=3))
    ax.add_artist(ab5)

    # Add DB Icon below App
    db_x = 17.5
    db_y = 1.5
    ax.annotate("", xy=(db_x, 2.5), xytext=(db_x, 3.0), arrowprops=dict(arrowstyle="<-", color="#7f8c8d", lw=2, linestyle=':'))
    
    ax.add_patch(patches.Ellipse((db_x, db_y+0.3), 1.2, 0.4, fc='#f1c40f', ec='#f39c12'))
    ax.add_patch(patches.Rectangle((db_x-0.6, db_y-0.4), 1.2, 0.7, fc='#f1c40f', ec='#f39c12'))
    ax.add_patch(patches.Ellipse((db_x, db_y-0.4), 1.2, 0.4, fc='#f1c40f', ec='#f39c12'))
    ax.text(db_x, db_y-1.0, "Session Database", ha='center', fontsize=10, fontweight='bold', color='#7f8c8d')


    plt.tight_layout()
    plt.savefig('project_pipeline.png', dpi=300, bbox_inches='tight')
    print("Saved visual pipeline with real images to project_pipeline.png")

if __name__ == "__main__":
    draw_project_pipeline_v2()
