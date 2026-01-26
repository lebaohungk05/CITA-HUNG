import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback

def draw_shadow_box(ax, x, y, text, w, h, fc, ec='#546e7a', fontsize=10):
    """
    Helper to draw a box with drop shadow.
    """
    shadow_offset = 0.15
    shadow_color = '#bdbdbd'
    
    # Shadow
    shadow = patches.FancyBboxPatch((x - w/2 + shadow_offset, y - h/2 - shadow_offset), w, h,
                                   boxstyle="round,pad=0.2", fc=shadow_color, ec='none', zorder=2)
    ax.add_patch(shadow)
    
    # Main Box
    main = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                 boxstyle="round,pad=0.2", fc=fc, ec=ec, lw=1.5, zorder=3)
    ax.add_patch(main)
    
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='normal', zorder=4, linespacing=1.4)

def draw_project_pipeline():
    try:
        # Setup figure - Landscape
        fig, ax = plt.subplots(figsize=(18, 10))
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 12)
        ax.axis('off')

        # COLORS (Pastel Professional)
        c_input = '#bbdefb'   # Light Blue
        c_proc = '#f5f5f5'    # Light Gray
        c_model = '#e1bee7'   # Light Purple (ResNet)
        c_out = '#c8e6c9'     # Light Green
        c_db = '#ffe0b2'      # Light Orange
        c_edge = '#455a64'
        
        arrow_args = dict(arrowstyle='->', color='#37474f', lw=2.0, mutation_scale=15)
        cy = 6.0 # Center Y

        # --- 1. INPUT ---
        draw_shadow_box(ax, 2.5, cy, "Webcam / Video\nStream", 3.0, 2.0, c_input, c_edge, fontsize=11)
        
        ax.annotate("", xy=(5.0, cy), xytext=(4.0, cy), arrowprops=arrow_args)

        # --- 2. FACE DETECTION ---
        draw_shadow_box(ax, 6.5, cy, "Face Detection\n(YuNet / Haar)", 3.0, 2.0, c_proc, c_edge)
        
        ax.annotate("", xy=(9.0, cy), xytext=(8.0, cy), arrowprops=arrow_args)

        # --- 3. PREPROCESSING ---
        # Container box
        draw_shadow_box(ax, 11.5, cy, "Preprocessing\n\n• Crop Face\n• Grayscale\n• Resize (48x48)\n• Normalize", 
                       3.5, 3.0, c_proc, c_edge, fontsize=9)
        
        ax.annotate("", xy=(14.2, cy), xytext=(13.3, cy), arrowprops=arrow_args)

        # --- 4. MODEL (THE CORE) ---
        # Larger box for importance
        draw_shadow_box(ax, 17.0, cy, "Emotion Recognition\nModel\n\n(SE-ResNet18 Mixup)", 
                       4.0, 2.5, c_model, c_edge, fontsize=11)
        
        # Details below model
        ax.text(17.0, cy - 1.8, "Input: (1, 48, 48)\nOutput: 7 Probabilities", ha='center', va='top', fontsize=9, style='italic', color='#546e7a')

        # Split arrows for output
        # 1. To Real-time Display
        ax.annotate("", xy=(21.0, cy + 1.5), xytext=(19.0, cy), arrowprops=arrow_args)
        
        # 2. To Analytics/DB
        ax.annotate("", xy=(21.0, cy - 1.5), xytext=(19.0, cy), arrowprops=arrow_args)

        # --- 5. OUTPUTS ---
        
        # Top: Visual Output
        draw_shadow_box(ax, 22.5, cy + 1.5, "Visual Output\n(Bounding Box\n+ Label)", 3.0, 2.0, c_out, c_edge, fontsize=10)
        
        # Bottom: Analytics
        draw_shadow_box(ax, 22.5, cy - 1.5, "Analytics System\n(Engagement Score\n+ Database)", 3.0, 2.0, c_db, c_edge, fontsize=10)

        # Connect Analytics to DB Icon (Symbolic)
        # Cylinder for DB
        db_x, db_y = 22.5, 2.0
        ax.add_patch(patches.Ellipse((db_x, db_y+0.3), 1.5, 0.4, fc='#ffcc80', ec=c_edge, zorder=5))
        ax.add_patch(patches.Rectangle((db_x-0.75, db_y-0.5), 1.5, 0.8, fc='#ffcc80', ec=c_edge, zorder=4))
        ax.add_patch(patches.Ellipse((db_x, db_y-0.5), 1.5, 0.4, fc='#ffcc80', ec=c_edge, zorder=5))
        ax.text(db_x, db_y-1.2, "Session Logs", ha='center', fontsize=9, fontweight='bold')
        
        ax.annotate("", xy=(22.5, 2.6), xytext=(22.5, 3.2), arrowprops=arrow_args)

        # Title
        plt.text(12.0, 11.0, "Figure 1. Real-time Emotion Recognition Pipeline", ha='center', fontsize=16, fontweight='bold', color='#263238')

        # Save
        plt.tight_layout()
        plt.savefig('project_pipeline.png', dpi=300, bbox_inches='tight')
        print("Figure saved to project_pipeline.png")
    
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    draw_project_pipeline()
