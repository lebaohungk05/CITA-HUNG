import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_shadow_box(ax, x, y, text, box_type='rect', fc='#e3f2fd', ec='#546e7a', w=None, h=None, fontsize=45):
    shadow_offset = 0.5
    shadow_color = '#bdbdbd'
    z_shadow, z_box, z_text = 2, 3, 4
    if w is None: w = 12.0  # Reduced width
    if h is None: h = 4.5   # Reduced height
    left, bottom = x - w/2, y - h/2
    
    if box_type == 'diamond':
        pts_shadow = [[x, y - h/2 - shadow_offset], [x + w/2 + shadow_offset, y], [x, y + h/2 - shadow_offset], [x - w/2 + shadow_offset, y]]
        ax.add_patch(patches.Polygon(pts_shadow, closed=True, fc=shadow_color, ec=None, zorder=z_shadow))
        pts_main = [[x, y - h/2], [x + w/2, y], [x, y + h/2], [x - w/2, y]]
        ax.add_patch(patches.Polygon(pts_main, closed=True, fc=fc, ec=ec, lw=4.0, zorder=z_box))
    else:
        shadow = patches.FancyBboxPatch((left + shadow_offset, bottom - shadow_offset), w, h, boxstyle="round,pad=0.3", fc=shadow_color, ec='none', zorder=z_shadow)
        ax.add_patch(shadow)
        main = patches.FancyBboxPatch((left, bottom), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=4.0, zorder=z_box)
        ax.add_patch(main)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', zorder=z_text, linespacing=1.3)

def draw_exact_flowchart():
    # HUGE SCALE FOR PAPER
    fig, ax = plt.subplots(figsize=(40, 60)) 
    ax.set_xlim(0, 60) 
    ax.set_ylim(-2, 65) 
    ax.axis('off')

    c_proc, c_dec, c_edge = '#e1f5fe', '#fff9c4', '#455a64'
    arrow_args = dict(arrowstyle='->', color='#263238', lw=6.0, mutation_scale=50)
    cx = 30.0

    # --- 1. INITIALIZATION ---
    # Shifted down slightly to give top margin
    ax.text(cx, 63.0, "Initialization", fontsize=70, fontweight='bold', ha='center', color='#263238')
    
    y_init = 59.0
    draw_shadow_box(ax, cx - 12.0, y_init, "Load ResNet18 Custom\n(Pretrained)", w=16.0, h=5.5, fc=c_proc, ec=c_edge)
    draw_shadow_box(ax, cx + 12.0, y_init, "Init Mixup Params\n(Alpha = 0.4)", w=16.0, h=5.5, fc=c_proc, ec=c_edge)
    
    # Connecting Init boxes to center line
    ax.plot([cx - 12.0, cx - 12.0], [y_init - 2.8, y_init - 4.0], color='#263238', lw=6.0)
    ax.plot([cx - 12.0, cx], [y_init - 4.0, y_init - 4.0], color='#263238', lw=6.0) 
    ax.plot([cx + 12.0, cx + 12.0], [y_init - 2.8, y_init - 4.0], color='#263238', lw=6.0)
    ax.plot([cx + 12.0, cx], [y_init - 4.0, y_init - 4.0], color='#263238', lw=6.0) 
    ax.annotate("", xy=(cx, y_init - 6.0), xytext=(cx, y_init - 4.0), arrowprops=arrow_args)

    # --- STAGE 1 ---
    # Moved down and compacted
    y_stage1_top = 51.0
    ax.plot([2, 58], [y_stage1_top, y_stage1_top], color='gray', ls=':', lw=3.0)
    ax.text(2.0, y_stage1_top + 1.0, "Stage 1: Standard Training", fontsize=75, fontweight='bold', color='#455a64', ha='left')
    
    y_s1_box_top = y_stage1_top - 2.0 # 49.0
    # Height approx 12 units
    rect_s1 = patches.FancyBboxPatch((4.0, y_s1_box_top - 12.0), 52.0, 12.0, boxstyle="round,pad=0.4", fc='#fafafa', ec='#cfd8dc', ls='--', lw=4.0, zorder=0)
    ax.add_patch(rect_s1)
    ax.text(5.5, y_s1_box_top - 1.0, "Standard Loop:\n• SGD + CosineLR\n• Mixup Augmentation", fontsize=62, color='#546e7a', ha='left', va='top')
    
    y_train_mixup = 46.0
    draw_shadow_box(ax, cx, y_train_mixup, "Train Epoch (Mixup)\nForward & Backward", w=18.0, h=5.5, fc=c_proc, ec=c_edge)
    ax.annotate("", xy=(cx, y_train_mixup - 4.0), xytext=(cx, y_train_mixup - 2.8), arrowprops=arrow_args)
    
    y_epoch200 = 40.0
    draw_shadow_box(ax, cx, y_epoch200, "Epoch < 200?", box_type='diamond', w=10.0, h=5.5, fc=c_dec, ec=c_edge)
    
    # Loop back YES
    ax.text(cx + 6.5, y_epoch200 + 1.0, "Yes", fontsize=60, fontweight='bold', color='#263238')
    ax.plot([cx + 5.0, cx + 18.0], [y_epoch200, y_epoch200], color='#263238', lw=6.0)
    ax.plot([cx + 18.0, cx + 18.0], [y_epoch200, y_train_mixup], color='#263238', lw=6.0)
    ax.annotate("", xy=(cx + 9.0, y_train_mixup), xytext=(cx + 18.0, y_train_mixup), arrowprops=arrow_args)
    
    # Continue NO
    ax.text(cx + 1.2, y_epoch200 - 3.5, "No", fontsize=60, fontweight='bold', color='#263238')
    ax.annotate("", xy=(cx, y_epoch200 - 5.0), xytext=(cx, y_epoch200 - 2.8), arrowprops=arrow_args)

    # --- TRANSITION ---
    y_switch = 33.0
    draw_shadow_box(ax, cx, y_switch, "Switch to SWA Mode\n(Init Averaged Model)", w=18.0, h=5.5, fc='#e0f7fa', ec=c_edge)
    ax.annotate("", xy=(cx, y_switch - 4.0), xytext=(cx, y_switch - 2.8), arrowprops=arrow_args)

    # --- STAGE 2 ---
    y_stage2_top = 28.0
    ax.plot([2, 58], [y_stage2_top, y_stage2_top], color='gray', ls=':', lw=3.0)
    ax.text(2.0, y_stage2_top + 1.0, "Stage 2: SWA Fine-tuning", fontsize=75, fontweight='bold', color='#455a64', ha='left')
    
    y_s2_box_top = y_stage2_top - 2.0 # 26.0
    # Height approx 16 units
    rect_s2 = patches.FancyBboxPatch((4.0, y_s2_box_top - 16.0), 52.0, 16.0, boxstyle="round,pad=0.4", fc='#fafafa', ec='#cfd8dc', ls='--', lw=4.0, zorder=0)
    ax.add_patch(rect_s2)
    ax.text(5.5, y_s2_box_top, "SWA Phase:\n• Constant LR\n• Update BN Stats", fontsize=62, color='#546e7a', ha='left', va='top')
    
    y_swa_step = 23.0
    draw_shadow_box(ax, cx, y_swa_step, "SWA Train Step\n& Update Weights", w=18.0, h=5.5, fc=c_proc, ec=c_edge)
    ax.annotate("", xy=(cx, y_swa_step - 4.0), xytext=(cx, y_swa_step - 2.8), arrowprops=arrow_args)
    
    y_val_acc = 17.0
    draw_shadow_box(ax, cx, y_val_acc, "Val Acc > Best?", box_type='diamond', w=10.0, h=5.5, fc=c_dec, ec=c_edge)
    
    # YES branch - Save Checkpoint
    ax.text(cx + 6.5, y_val_acc + 1.0, "Yes", fontsize=60, fontweight='bold', color='#263238')
    ax.annotate("", xy=(cx + 17.0, y_val_acc), xytext=(cx + 5.0, y_val_acc), arrowprops=arrow_args)
    
    draw_shadow_box(ax, cx + 20.0, y_val_acc, "Save Best\nCheckpoint", w=12.0, h=4.5, fc='#dcedc8', ec=c_edge)
    
    # Loop back from Checkpoint
    ax.plot([cx + 20.0, cx + 20.0], [y_val_acc - 2.3, y_val_acc - 3.5], color='#263238', lw=6.0) 
    ax.plot([cx + 20.0, cx], [y_val_acc - 3.5, y_val_acc - 3.5], color='#263238', lw=6.0)
    
    # NO branch
    ax.text(cx + 1.2, y_val_acc - 2.5, "No", fontsize=60, fontweight='bold', color='#263238')
    ax.plot([cx, cx], [y_val_acc - 2.8, y_val_acc - 3.5], color='#263238', lw=6.0) 
    ax.annotate("", xy=(cx, y_val_acc - 5.5), xytext=(cx, y_val_acc - 3.5), arrowprops=arrow_args)
    
    y_epoch240 = 10.0
    draw_shadow_box(ax, cx, y_epoch240, "Epoch == 240?", box_type='diamond', w=10.0, h=5.5, fc=c_dec, ec=c_edge)
    
    # Loop back NO
    ax.text(cx - 6.5, y_epoch240 + 1.0, "No", fontsize=60, fontweight='bold', color='#263238')
    ax.plot([cx - 5.0, cx - 22.0], [y_epoch240, y_epoch240], color='#263238', lw=6.0)
    ax.plot([cx - 22.0, cx - 22.0], [y_epoch240, y_swa_step], color='#263238', lw=6.0)
    ax.annotate("", xy=(cx - 9.0, y_swa_step), xytext=(cx - 22.0, y_swa_step), arrowprops=arrow_args)
    
    # YES - Finish
    ax.text(cx + 1.2, y_epoch240 - 3.5, "Yes", fontsize=60, fontweight='bold', color='#263238')
    ax.annotate("", xy=(cx, y_epoch240 - 6.0), xytext=(cx, y_epoch240 - 2.8), arrowprops=arrow_args)

    y_final = 2.0 # Much higher than -5.5
    draw_shadow_box(ax, cx, y_final, "Final Optimized Model\n(ResNet Mixup 72.5%)", w=20.0, h=5.5, fc='#c5e1a5', ec=c_edge)

    plt.tight_layout()
    plt.savefig('Figure2_Proposed_Training_Strategy.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    draw_exact_flowchart()