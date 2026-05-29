import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

C = {
    'data':       '#E3F2FD',
    'preproc':    '#E8F5E9',
    'augment':    '#FFF3E0',
    'norm':       '#F3E5F5',
    'split':      '#E0F7FA',
    'model_cnn':  '#BBDEFB',
    'model_res':  '#C8E6C9',
    'model_eff':  '#FFE0B2',
    'train':      '#FCE4EC',
    'eval':       '#F5F5F5',
    'output':     '#FFCDD2',
    'border':     '#37474F',
    'arrow':      '#455A64',
    'phase_bg':   '#FAFAFA',
    'head_data':  '#1565C0',
    'head_pre':   '#2E7D32',
    'head_model': '#C62828',
    'head_eval':  '#4A148C',
}


def box(ax, cx, cy, w, h, text, color, fontsize=8, bold=False, lw=1.0, ec=None):
    ec = ec or C['border']
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.03", facecolor=color,
                           edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=4, linespacing=1.3)

def arr(ax, x1, y1, x2, y2, color=None):
    color = color or C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3),
                zorder=2)

def arr_curved(ax, x1, y1, x2, y2, color=None, rad=0.3):
    color = color or C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=2)

def phase_label(ax, x, y, text, color):
    ax.text(x, y, text, fontsize=11, fontweight='bold', color=color,
            ha='center', va='center', zorder=5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=color, linewidth=1.5, alpha=0.95))

def section_bg(ax, x, y, w, h, color='#F5F5F5'):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor='#BDBDBD',
                           linewidth=0.8, zorder=0, alpha=0.4)
    ax.add_patch(patch)


fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(-1, 21)
ax.set_ylim(-0.5, 14)
ax.axis('off')
ax.set_title('CIFAR-10 Classification — Complete Workflow',
             fontsize=16, fontweight='bold', pad=20)

# ================================================================
# PHASE 1: DATA LOADING & PREPROCESSING (top section)
# ================================================================
section_bg(ax, -0.5, 10.5, 21, 3.2, '#E3F2FD')
phase_label(ax, 2.5, 13.2, 'Phase 1: Data Loading & Preprocessing', C['head_data'])

# Row 1: Data loading
bw, bh = 2.2, 0.8

box(ax, 1.5, 12.2, 2.5, bh, 'CIFAR-10 Dataset\n(.mat files)\n50K train + 10K test', C['data'], bold=True)
arr(ax, 2.75, 12.2, 4.3, 12.2)
box(ax, 5.5, 12.2, bw, bh, 'Load & Reshape\n(N, 32, 32, 3)\nuint8 [0-255]', C['data'])
arr(ax, 6.6, 12.2, 7.8, 12.2)
box(ax, 9.0, 12.2, bw, bh, 'Gaussian Blur\n(3×3, σ=0.5)\nDenoising', C['preproc'])
arr(ax, 10.1, 12.2, 11.3, 12.2)
box(ax, 12.5, 12.2, bw, bh, 'CLAHE\nclipLimit=2.0\ntileGrid=4×4', C['preproc'])
arr(ax, 13.6, 12.2, 14.8, 12.2)
box(ax, 16.0, 12.2, bw, bh, 'Unsharp Mask\nSharpening\n(1.5× weight)', C['preproc'])

# Arrow down to augmentation
arr(ax, 16.0, 11.8, 16.0, 11.2)

# Row 2: Augmentation + Normalization
box(ax, 16.0, 10.8, 2.8, 0.7, '50K Processed Images', C['preproc'], bold=True, fontsize=7.5)

arr(ax, 14.6, 10.8, 13.2, 10.8)
box(ax, 11.8, 10.8, 2.6, 0.7, 'Data Augmentation\n+25K (balanced per class)', C['augment'], bold=True, fontsize=7.5)

# Augmentation details box
box(ax, 8.5, 10.8, 3.0, 0.7, 'Flip | Rotation (±15°)\nRandom Crop | Brightness\nCutout (10px)', C['augment'], fontsize=7)

arr(ax, 10.5, 10.8, 11.8 - 1.3, 10.8)

arr(ax, 8.5 - 1.5, 10.8, 5.9, 10.8)
box(ax, 4.7, 10.8, 2.2, 0.7, '75K Total\nImages', C['augment'], bold=True, fontsize=8)

arr(ax, 3.6, 10.8, 2.8, 10.8)
box(ax, 1.7, 10.8, 2.0, 0.7, 'Normalize\n÷255 → [0,1]\nStandardize', C['norm'], fontsize=7.5)

# ================================================================
# PHASE 2: TRAIN/VAL/TEST SPLIT
# ================================================================
section_bg(ax, -0.5, 8.8, 21, 1.5, '#E0F7FA')
phase_label(ax, 2.5, 9.9, 'Phase 2: Data Split', C['head_pre'])

arr(ax, 1.7, 10.45, 1.7, 9.65)

box(ax, 4.0, 9.2, 2.5, 0.7, 'Train: 67,500\n(90%, stratified)', C['split'], bold=True, fontsize=8)
box(ax, 8.0, 9.2, 2.5, 0.7, 'Validation: 7,500\n(10%, stratified)', C['split'], bold=True, fontsize=8)
box(ax, 12.5, 9.2, 2.5, 0.7, 'Test: 10,000\n(separate set)', C['split'], bold=True, fontsize=8)

arr(ax, 2.7, 9.2, 4.0 - 1.25, 9.2)
arr(ax, 4.0 + 1.25, 9.2, 8.0 - 1.25, 9.2)
arr(ax, 8.0 + 1.25, 9.2, 12.5 - 1.25, 9.2)

# Save NPZ
box(ax, 17.0, 9.2, 2.8, 0.7, 'Save → .npz\ncifar10_preprocessed.npz\n(381.8 MB)', C['data'], fontsize=7)
arr(ax, 12.5 + 1.25, 9.2, 17.0 - 1.4, 9.2)

# ================================================================
# PHASE 3: MODEL TRAINING (three parallel tracks)
# ================================================================
section_bg(ax, -0.5, 3.5, 21, 5.0, '#FFF8E1')
phase_label(ax, 2.5, 8.1, 'Phase 3: Model Training', C['head_model'])

# --- Track A: Custom CNN ---
track_a_x = 3.5
box(ax, track_a_x, 7.2, 3.8, 0.6, 'Custom CNN (from scratch)', C['model_cnn'], bold=True, fontsize=9, lw=1.5, ec='#1565C0')

box(ax, track_a_x, 6.3, 3.4, 0.55, 'Block 1: Conv(64)×2 + BN + ReLU\nMaxPool → 16×16×64, Drop(0.4)', C['model_cnn'], fontsize=6.5)
arr(ax, track_a_x, 6.9, track_a_x, 6.58)

box(ax, track_a_x, 5.5, 3.4, 0.55, 'Block 2: Conv(128)×2 + BN + ReLU\nMaxPool → 8×8×128, Drop(0.5)', C['model_cnn'], fontsize=6.5)
arr(ax, track_a_x, 6.02, track_a_x, 5.78)

box(ax, track_a_x, 4.7, 3.4, 0.55, 'Block 3: Conv(256)×2 + BN + ReLU\nMaxPool → 4×4×256, Drop(0.5)', C['model_cnn'], fontsize=6.5)
arr(ax, track_a_x, 5.22, track_a_x, 4.98)

box(ax, track_a_x, 3.9, 3.4, 0.5, 'Flatten → Dense(512) → BN → ReLU\nDrop(0.6) → Dense(10, softmax)', C['model_cnn'], fontsize=6.5)
arr(ax, track_a_x, 4.42, track_a_x, 4.15)

# Training details
ax.text(track_a_x, 3.35, 'SGD (lr=0.01, CosineDecay)\n100 epochs, BS=128\nEarlyStopping (patience=10)\n3.25M params',
        fontsize=6.5, ha='center', va='center', color='#1565C0',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#90CAF9', alpha=0.9))

# Input arrow from data split
arr_curved(ax, 4.0, 8.85, track_a_x, 7.5, color='#1565C0', rad=-0.15)

# --- Track B: ResNet50V2 ---
track_b_x = 10.0
box(ax, track_b_x, 7.2, 3.8, 0.6, 'ResNet50V2 (Transfer Learning)', C['model_res'], bold=True, fontsize=9, lw=1.5, ec='#2E7D32')

box(ax, track_b_x, 6.3, 3.4, 0.55, 'Resize 32×32 → 224×224\nResNet preprocess_input', C['model_res'], fontsize=7)
arr(ax, track_b_x, 6.9, track_b_x, 6.58)

box(ax, track_b_x, 5.45, 3.4, 0.65, 'ResNet50V2 Backbone\n(ImageNet pretrained, 23.6M params)', '#E1BEE7', fontsize=7, lw=1.5, ec='#6A1B9A')
arr(ax, track_b_x, 6.02, track_b_x, 5.78)

box(ax, track_b_x, 4.6, 3.4, 0.55, 'GAP → Dropout(0.3)\n→ Dense(10, softmax)', C['model_res'], fontsize=7)
arr(ax, track_b_x, 5.12, track_b_x, 4.88)

# Training details
ax.text(track_b_x, 3.75, 'Phase 1: Freeze → Adam (lr=0.001)\n     Train head, 10 epochs\nPhase 2: Unfreeze → SGD (lr=0.001)\n     Fine-tune all, 20 epochs',
        fontsize=6.5, ha='center', va='center', color='#2E7D32',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#A5D6A7', alpha=0.9))

arr_curved(ax, 8.0, 8.85, track_b_x, 7.5, color='#2E7D32', rad=0.0)

# --- Track C: EfficientNetB0 ---
track_c_x = 16.5
box(ax, track_c_x, 7.2, 3.8, 0.6, 'EfficientNetB0 (Transfer Learning)', C['model_eff'], bold=True, fontsize=9, lw=1.5, ec='#E65100')

box(ax, track_c_x, 6.3, 3.4, 0.55, 'Resize 32×32 → 224×224\nEfficientNet preprocess_input', C['model_eff'], fontsize=7)
arr(ax, track_c_x, 6.9, track_c_x, 6.58)

box(ax, track_c_x, 5.45, 3.4, 0.65, 'EfficientNetB0 Backbone\n(ImageNet pretrained, 4.05M params)\nCompound Scaling', '#E1BEE7', fontsize=7, lw=1.5, ec='#6A1B9A')
arr(ax, track_c_x, 6.02, track_c_x, 5.78)

box(ax, track_c_x, 4.6, 3.4, 0.55, 'GAP → Dropout(0.3)\n→ Dense(10, softmax)', C['model_eff'], fontsize=7)
arr(ax, track_c_x, 5.12, track_c_x, 4.88)

ax.text(track_c_x, 3.75, 'Phase 1: Freeze → Adam (lr=0.001)\n     Train head, 10 epochs\nPhase 2: Unfreeze → SGD (lr=0.001)\n     Fine-tune all, 20 epochs',
        fontsize=6.5, ha='center', va='center', color='#E65100',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#FFCC80', alpha=0.9))

arr_curved(ax, 12.5, 8.85, track_c_x, 7.5, color='#E65100', rad=0.15)

# ================================================================
# PHASE 4: EVALUATION & COMPARISON
# ================================================================
section_bg(ax, -0.5, -0.3, 21, 3.5, '#F3E5F5')
phase_label(ax, 2.5, 2.8, 'Phase 4: Evaluation & Comparison', C['head_eval'])

# Evaluation boxes
eval_y = 1.8
box(ax, track_a_x, eval_y, 3.2, 0.7, 'Test Evaluation\nAccuracy: 86.89%', C['eval'], fontsize=7.5, lw=1.2, ec='#1565C0')
arr(ax, track_a_x, 3.05, track_a_x, eval_y + 0.35)

box(ax, track_b_x, eval_y, 3.2, 0.7, 'Test Evaluation\nAccuracy: 94.30%', C['eval'], fontsize=7.5, lw=1.2, ec='#2E7D32')
arr(ax, track_b_x, 3.05, track_b_x, eval_y + 0.35)

box(ax, track_c_x, eval_y, 3.2, 0.7, 'Test Evaluation\nAccuracy: 83.75%', C['eval'], fontsize=7.5, lw=1.2, ec='#E65100')
arr(ax, track_c_x, 3.05, track_c_x, eval_y + 0.35)

# Final comparison
comp_y = 0.5
box(ax, 10.0, comp_y, 8.0, 0.7,
    'Model Comparison: Training Curves | Confusion Matrices | Accuracy Bar Chart',
    C['output'], bold=True, fontsize=9, lw=1.5, ec='#C62828')

arr(ax, track_a_x, eval_y - 0.35, 6.5, comp_y + 0.35)
arr(ax, track_b_x, eval_y - 0.35, 10.0, comp_y + 0.35)
arr(ax, track_c_x, eval_y - 0.35, 13.5, comp_y + 0.35)

# ================================================================
# Legend
# ================================================================
legend_elements = [
    mpatches.Patch(facecolor=C['data'], edgecolor=C['border'], label='Data Loading'),
    mpatches.Patch(facecolor=C['preproc'], edgecolor=C['border'], label='Preprocessing'),
    mpatches.Patch(facecolor=C['augment'], edgecolor=C['border'], label='Augmentation'),
    mpatches.Patch(facecolor=C['norm'], edgecolor=C['border'], label='Normalization'),
    mpatches.Patch(facecolor=C['split'], edgecolor=C['border'], label='Data Split'),
    mpatches.Patch(facecolor=C['model_cnn'], edgecolor=C['border'], label='Custom CNN'),
    mpatches.Patch(facecolor=C['model_res'], edgecolor=C['border'], label='ResNet50V2'),
    mpatches.Patch(facecolor=C['model_eff'], edgecolor=C['border'], label='EfficientNetB0'),
    mpatches.Patch(facecolor='#E1BEE7', edgecolor='#6A1B9A', label='Pretrained Backbone'),
    mpatches.Patch(facecolor=C['output'], edgecolor=C['border'], label='Final Output'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=7.5, ncol=5,
          framealpha=0.95, bbox_to_anchor=(-0.02, -0.04))

plt.tight_layout()
plt.savefig('/Users/mjabed3834/Documents/UNIMIB/Second Semester/Deep Learning/cifar-10-batches-mat/TensorFlow/Figure/workflow_diagram.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Workflow diagram saved: TensorFlow/Figure/workflow_diagram.png")
