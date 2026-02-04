from graphviz import Digraph
import os

# Cấu hình đường dẫn
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILENAME = 'Proposed_v2'

def draw_pipeline():
    dot = Digraph(comment='Training Pipeline', format='png')
    dot.attr(rankdir='TB', size='10,12', ratio='fill')
    
    # Global node settings
    dot.attr('node', shape='rect', style='filled', fontname='Times-Roman', fontsize='12')
    
    # --- BLOCK 1: INITIALIZATION ---
    with dot.subgraph(name='cluster_0') as c:
        c.attr(style='invis')
        c.node('Start', 'Load SE-ResNet18\n(Pretrained)', fillcolor='lightblue')
        c.node('InitMixup', 'Init Mixup Params\n(alpha = 0.6)', fillcolor='lightblue')
    
    # --- BLOCK 2: STAGE 1 (Standard Training) ---
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Stage 1: Base Training (Convergence)', fontsize='14', style='dashed', color='gray')
        c.node('TrainStep1', 'Train Epoch (Mixup)\nSGD + CosineLR', fillcolor='lightblue')
        c.node('CheckEpoch1', 'Epoch < 230?', shape='diamond', fillcolor='lightyellow')
        
    # --- BLOCK 3: STAGE 2 (SWA) ---
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Stage 2: SWA Generalization', fontsize='14', style='dashed', color='gray')
        c.node('SwitchSWA', 'Switch to SWA Mode\n(LR = 0.001)', fillcolor='lightblue')
        c.node('TrainStep2', 'SWA Train Step\n& Update Weights', fillcolor='lightblue')
        c.node('CheckEpoch2', 'Epoch = 265?', shape='diamond', fillcolor='lightyellow')

    # --- BLOCK 4: STAGE 3 (Fine-tuning) ---
    with dot.subgraph(name='cluster_3') as c:
        c.attr(label='Stage 3: Fine-tuning (Refinement)', fontsize='14', style='dashed', color='gray')
        c.node('DisableMixup', 'Disable Mixup\nUse Clean Data', fillcolor='orange')
        c.node('FreezeTrain', 'Step 3a: Freeze Backbone\nTrain Classifier (LR 1e-3)', fillcolor='orange')
        c.node('UnfreezeTrain', 'Step 3b: Unfreeze All\nTrain (LR 1e-5)', fillcolor='orange')

    # --- OUTPUT ---
    dot.node('Final', 'Final Optimized Model\n(Acc 72.72%)', shape='rect', style='filled,rounded', fillcolor='lightgreen', fontsize='14')

    # --- EDGES ---
    dot.edge('Start', 'TrainStep1')
    dot.edge('InitMixup', 'TrainStep1')
    
    # Loop Stage 1
    dot.edge('TrainStep1', 'CheckEpoch1')
    dot.edge('CheckEpoch1', 'TrainStep1', label='Yes')
    dot.edge('CheckEpoch1', 'SwitchSWA', label='No')
    
    # Loop Stage 2
    dot.edge('SwitchSWA', 'TrainStep2')
    dot.edge('TrainStep2', 'CheckEpoch2')
    dot.edge('CheckEpoch2', 'TrainStep2', label='No')
    dot.edge('CheckEpoch2', 'DisableMixup', label='Yes')
    
    # Linear Stage 3
    dot.edge('DisableMixup', 'FreezeTrain')
    dot.edge('FreezeTrain', 'UnfreezeTrain')
    dot.edge('UnfreezeTrain', 'Final')

    # Render
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    dot.render(output_path, view=False)
    print(f"Pipeline diagram saved to: {output_path}.png")

if __name__ == '__main__':
    try:
        draw_pipeline()
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure 'graphviz' is installed on your system (both python package and binary).")
        print("Command: pip install graphviz")
        print("Download: https://graphviz.org/download/")
