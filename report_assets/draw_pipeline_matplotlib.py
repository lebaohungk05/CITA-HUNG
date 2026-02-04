import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILENAME = 'Proposed_v2_matplotlib.png'

def draw_box(ax, x, y, w, h, text, color='#E0E0E0', edge_color='black'):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                  linewidth=1, edgecolor=edge_color, facecolor=color)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
    return x + w/2, y  # Return bottom center for connection

def draw_diamond(ax, x, y, size, text, color='#FFFFCC'):
    # Diamond shape using Polygon
    pts = [[x, y + size], [x + size*1.5, y], [x, y - size], [x - size*1.5, y]]
    poly = patches.Polygon(pts, closed=True, edgecolor='black', facecolor=color)
    ax.add_patch(poly)
    ax.text(x, y, text, ha='center', va='center', fontsize=9)
    return x, y - size

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color='black', lw=1.5))

def main():
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Coordinates
    center_x = 5
    current_y = 13.5
    
    # --- HEADER ---
    _, y_load = draw_box(ax, 1, current_y - 1, 3, 1, "Load SE-ResNet18\n(Pretrained)", color='#BDE0FE')
    _, y_init = draw_box(ax, 6, current_y - 1, 3, 1, "Init Mixup Params\n(alpha = 0.6)", color='#BDE0FE')
    
    current_y -= 1.5
    
    # Merge point
    draw_arrow(ax, 2.5, y_load, center_x, current_y)
    draw_arrow(ax, 7.5, y_init, center_x, current_y)
    
    # --- STAGE 1 ---
    # Container
    rect1 = patches.Rectangle((1, 8.5), 8, 4.5, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect1)
    ax.text(1.2, 12.7, "Stage 1: Base Training (Convergence)", fontsize=11, color='gray', fontweight='bold')
    
    current_y -= 0.5
    _, y_train1 = draw_box(ax, 3, current_y - 1, 4, 1, "Train Epoch (Mixup)\nSGD + CosineLR", color='#BDE0FE')
    
    current_y -= 1.5
    x_check1, y_check1 = draw_diamond(ax, center_x, current_y, 0.5, "Epoch < 230?")
    
    # Loop arrow
    draw_arrow(ax, center_x, y_train1, center_x, current_y + 0.5) # Down to check
    # Yes loop
    ax.annotate("", xy=(7, 11.5), xytext=(7, 10.5), arrowprops=dict(arrowstyle="-", color='black')) # Up line
    ax.text(6.5, 10.5, "Yes", fontsize=10)
    draw_arrow(ax, 6.5, 10.5, 7, 10.5) # Right from check to line
    draw_arrow(ax, 7, 11.5, 7, 11.5) # Top right
    draw_arrow(ax, 7, 11.5, 7, 11.5) 
    ax.arrow(center_x + 1.5, current_y, 1.5, 0, head_width=0, color='black') # Right
    ax.arrow(center_x + 3, current_y, 0, 1.5, head_width=0, color='black') # Up
    ax.arrow(center_x + 3, current_y + 1.5, -1, 0, head_width=0.1, color='black') # Back to Train
    
    # --- STAGE 2 ---
    rect2 = patches.Rectangle((1, 4.5), 8, 3.5, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect2)
    ax.text(1.2, 7.7, "Stage 2: SWA Generalization", fontsize=11, color='gray', fontweight='bold')
    
    # No from Stage 1
    current_y -= 1.0
    draw_arrow(ax, center_x, y_check1, center_x, current_y)
    ax.text(center_x + 0.2, y_check1 - 0.3, "No", fontsize=10)
    
    _, y_swa_init = draw_box(ax, 3, current_y - 1, 4, 1, "Switch to SWA Mode\n(LR = 0.001)", color='#BDE0FE')
    
    current_y -= 1.5
    _, y_train2 = draw_box(ax, 3, current_y - 1, 4, 1, "SWA Train Step\n& Update Weights", color='#BDE0FE')
    draw_arrow(ax, center_x, y_swa_init, center_x, current_y)
    
    current_y -= 1.2
    x_check2, y_check2 = draw_diamond(ax, center_x, current_y, 0.5, "Epoch = 265?")
    draw_arrow(ax, center_x, y_train2, center_x, current_y + 0.5)
    
    # No loop for Stage 2
    ax.arrow(center_x + 1.5, current_y, 1.5, 0, head_width=0, color='black') # Right
    ax.arrow(center_x + 3, current_y, 0, 1.5, head_width=0, color='black') # Up
    ax.arrow(center_x + 3, current_y + 1.5, -1, 0, head_width=0.1, color='black') # Back
    ax.text(center_x + 1.6, current_y + 0.2, "No", fontsize=10)

    # --- STAGE 3 (FINE TUNING) ---
    rect3 = patches.Rectangle((1, 0.5), 8, 3.5, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect3)
    ax.text(1.2, 3.7, "Stage 3: Fine-tuning (Refinement)", fontsize=11, color='gray', fontweight='bold')
    
    # Yes from Stage 2
    current_y -= 1.0
    draw_arrow(ax, center_x, y_check2, center_x, current_y)
    ax.text(center_x + 0.2, y_check2 - 0.3, "Yes", fontsize=10)
    
    _, y_disable = draw_box(ax, 1.5, current_y - 0.8, 3, 0.8, "Disable Mixup", color='#FFCC99')
    _, y_freeze = draw_box(ax, 5.5, current_y - 0.8, 3, 0.8, "Freeze Backbone\nTrain FC (1e-3)", color='#FFCC99')
    
    draw_arrow(ax, center_x, current_y, 3, current_y) # Split left
    draw_arrow(ax, 3, current_y, 3, current_y - 0.2)
    draw_arrow(ax, center_x, current_y, 7, current_y) # Split right
    draw_arrow(ax, 7, current_y, 7, current_y - 0.2)
    
    # Connection internal stage 3
    draw_arrow(ax, 3, y_disable, 7, y_disable + 0.4) # Not quite right layout
    # Re-draw linear flow for Stage 3 to be clearer
    
    # CLEAR STAGE 3 AREA TO REDRAW LINEAR
    rect3.set_visible(False) # Hide old rect
    rect3_new = patches.Rectangle((1, 0.2), 8, 3.8, linewidth=1, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect3_new)
    ax.text(1.2, 3.7, "Stage 3: Fine-tuning Strategy", fontsize=11, color='gray', fontweight='bold')

    current_y = 3.2
    _, y_ft1 = draw_box(ax, 3, current_y - 0.8, 4, 0.8, "Disable Mixup & Freeze\nTrain Head (LR 1e-3)", color='#FFCC99')
    
    current_y -= 1.2
    _, y_ft2 = draw_box(ax, 3, current_y - 0.8, 4, 0.8, "Unfreeze All Layers\nTrain (LR 1e-5)", color='#FFCC99')
    
    draw_arrow(ax, center_x, y_check2, center_x, y_ft1 + 0.8) # From Stage 2
    draw_arrow(ax, center_x, y_ft1, center_x, y_ft1 - 0.4) # Down 1
    
    current_y -= 1.5
    # --- FINAL OUTPUT ---
    draw_box(ax, 2.5, current_y - 1, 5, 1, "Final Model (Mix-SEResNet)\nAccuracy: 72.72%", color='#90EE90')
    draw_arrow(ax, center_x, y_ft2, center_x, current_y)

    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=300)
    print(f"Saved: {output_path}")

if __name__ == '__main__':
    main()
