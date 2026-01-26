import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_flowchart():
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Styles
    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e1f5fe', edgecolor='#0277bd', linewidth=1.5)
    decision_props = dict(boxstyle='darrow,pad=0.3', facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=1.5)
    start_props = dict(boxstyle='round,pad=0.5', facecolor='#e0e0e0', edgecolor='#424242', linewidth=1.5)
    final_props = dict(boxstyle='round,pad=0.5', facecolor='#c8e6c9', edgecolor='#2e7d32', linewidth=1.5)
    
    arrow_props = dict(arrowstyle='->', lw=1.5, color='#37474f')
    text_props = dict(ha='center', va='center', fontsize=10, fontweight='normal', color='black')

    # --- Nodes ---
    
    # 1. Initialization
    ax.text(5, 13, "Initialization", ha='center', fontsize=12, fontweight='bold')
    rect_init = patches.FancyBboxPatch((3.0, 11.5), 4.0, 1.0, **start_props)
    ax.add_patch(rect_init)
    ax.text(5, 12.0, "Load Custom ResNet18\nInit Mixup (alpha=0.4)", **text_props)

    # Arrow Init -> Stage 1
    ax.annotate('', xy=(5, 10.5), xytext=(5, 11.5), arrowprops=arrow_props)

    # Group: Stage 1
    rect_s1_bg = patches.Rectangle((1.5, 7.5), 7.0, 3.5, fill=False, linestyle='--', edgecolor='gray', lw=1)
    ax.add_patch(rect_s1_bg)
    ax.text(2.0, 10.8, "Stage 1: Standard Training", fontsize=9, fontweight='bold', color='gray')

    # 2. Stage 1: Standard Training
    rect_stage1 = patches.FancyBboxPatch((2.5, 9.0), 5.0, 1.5, **box_props)
    ax.add_patch(rect_stage1)
    ax.text(5, 9.75, "Standard Training Loop\n(SGD + CosineAnnealingLR)\nMixup Augmentation", **text_props)

    # 3. Decision: Epoch >= 200?
    # Using a diamond shape manually or via boxstyle 'darrow' (which is horizontal usually).
    # Let's use a rotated square for diamond visual
    # Or just 'diamond' boxstyle if available? 'darrow' is not diamond.
    # Matplotlib's 'boxstyle' doesn't have a perfect diamond. I'll use a patch.
    
    # Decision Diamond Center at (5, 8.0)
    diamond = patches.RegularPolygon((5, 8.0), numVertices=4, radius=0.8, orientation=0, facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=1.5)
    # Note: RegularPolygon orientation might need tweaking. 0 is usually point up?
    # Actually let's use a simple text box with 'diamond' shape if possible, or just draw it.
    # bbox_props of text can be 'darrow' but 'diamond' is not standard.
    # I'll stick to a Box that says "Check Epoch"
    
    ax.text(5, 8.0, "Epoch >= 200?", ha='center', va='center', fontsize=9, bbox=dict(boxstyle="square,pad=0.5", fc='#fff9c4', ec='#fbc02d'))

    # Arrow Stage 1 -> Decision
    ax.annotate('', xy=(5, 8.4), xytext=(5, 9.0), arrowprops=arrow_props)

    # Loop Back (No)
    # Right side loop
    ax.annotate('No', xy=(6.5, 8.0), xytext=(8.0, 8.0), arrowprops=arrow_props)
    ax.annotate('', xy=(8.0, 9.75), xytext=(8.0, 4.0), arrowprops=dict(arrowstyle='-', lw=1.5, color='#37474f'))
    ax.annotate('', xy=(7.5, 9.75), xytext=(8.0, 9.75), arrowprops=arrow_props)

    # Arrow Decision -> Stage 2 (Yes)
    ax.text(5.2, 7.2, "Yes", fontsize=9)
    ax.annotate('', xy=(5, 6.5), xytext=(5, 7.6), arrowprops=arrow_props)

    # Group: Stage 2
    rect_s2_bg = patches.Rectangle((1.5, 3.0), 7.0, 4.0, fill=False, linestyle='--', edgecolor='gray', lw=1)
    ax.add_patch(rect_s2_bg)
    ax.text(2.0, 6.8, "Stage 2: SWA Fine-tuning", fontsize=9, fontweight='bold', color='gray')

    # 4. Stage 2: SWA
    rect_stage2 = patches.FancyBboxPatch((2.5, 5.0), 5.0, 1.5, **box_props)
    ax.add_patch(rect_stage2)
    ax.text(5, 5.75, "SWA Training Phase\n(Stochastic Weight Averaging)\nUpdate BN Statistics", **text_props)

    # 5. Decision: Epoch == 240?
    ax.text(5, 4.0, "Epoch == 240?", ha='center', va='center', fontsize=9, bbox=dict(boxstyle="square,pad=0.5", fc='#fff9c4', ec='#fbc02d'))

    # Arrow Stage 2 -> Decision
    ax.annotate('', xy=(5, 4.4), xytext=(5, 5.0), arrowprops=arrow_props)

    # Loop Back (No)
    ax.annotate('No', xy=(6.5, 4.0), xytext=(8.0, 4.0), arrowprops=arrow_props)
    ax.annotate('', xy=(8.0, 5.75), xytext=(8.0, 4.0), arrowprops=dict(arrowstyle='-', lw=1.5, color='#37474f'))
    ax.annotate('', xy=(7.5, 5.75), xytext=(8.0, 5.75), arrowprops=arrow_props)

    # Arrow Decision -> End (Yes)
    ax.text(5.2, 3.2, "Yes", fontsize=9)
    ax.annotate('', xy=(5, 2.5), xytext=(5, 3.6), arrowprops=arrow_props)

    # 6. Final
    rect_final = patches.FancyBboxPatch((3.0, 1.0), 4.0, 1.5, **final_props)
    ax.add_patch(rect_final)
    ax.text(5, 1.75, "Final Optimized Model\n(ResNet Mixup)\nBest Acc: ~72.5%", **text_props)

    # Title
    plt.title("ResNet Mixup Training Pipeline\n(Adapted from Fig. 2 Style)", fontsize=14, pad=20)
    
    # Save
    plt.tight_layout()
    plt.savefig('resnet_mixup_pipeline_fig2.png', dpi=300, bbox_inches='tight')
    print("Flowchart saved to resnet_mixup_pipeline_fig2.png")

if __name__ == "__main__":
    draw_flowchart()
