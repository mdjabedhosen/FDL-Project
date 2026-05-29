import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

FIG_DIR = '/Users/mjabed3834/Documents/UNIMIB/Second Semester/Deep Learning/cifar-10-batches-mat/TensorFlow/Figure'

C = {
    'data':     '#E3F2FD',
    'denoise':  '#E8F5E9',
    'clahe':    '#FFF3E0',
    'sharpen':  '#FCE4EC',
    'augment':  '#F3E5F5',
    'norm':     '#E0F7FA',
    'split':    '#FFF9C4',
    'save':     '#FFCDD2',
    'border':   '#37474F',
    'arrow':    '#455A64',
}

def box(ax, cx, cy, w, h, text, color, fontsize=9, bold=False, lw=1.2, ec=None):
    ec = ec or C['border']
    patch = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.04", facecolor=color,
                           edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(patch)
    weight = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=4, linespacing=1.3)

def arr_down(ax, x, y1, y2, color=None):
    color = color or C['arrow']
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=2)

def arr_right(ax, x1, x2, y, color=None):
    color = color or C['arrow']
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=2)

def arr_diag(ax, x1, y1, x2, y2, color=None):
    color = color or C['arrow']
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3), zorder=2)

def step_num(ax, x, y, num):
    ax.text(x, y, str(num), ha='center', va='center', fontsize=11,
            fontweight='bold', color='white', zorder=6,
            bbox=dict(boxstyle='circle,pad=0.3', facecolor='#1565C0',
                      edgecolor='white', linewidth=2))

def section_bg(ax, x, y, w, h, color='#F5F5F5', ec='#BDBDBD'):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor=ec,
                           linewidth=0.8, zorder=0, alpha=0.35)
    ax.add_patch(patch)


# ================================================================
fig, ax = plt.subplots(1, 1, figsize=(16, 18))
ax.set_xlim(0, 16)
ax.set_ylim(0, 18)
ax.axis('off')
fig.suptitle('CIFAR-10 Data Preprocessing Pipeline',
             fontsize=18, fontweight='bold', y=0.985)

cx = 8.0

# ================================================================
# STAGE 1: DATA LOADING
# ================================================================
y = 17.0
section_bg(ax, 1, y - 0.7, 14, 1.5, '#E3F2FD', '#90CAF9')
ax.text(cx, y + 0.55, 'Stage 1: Data Loading', fontsize=12, fontweight='bold',
        ha='center', color='#1565C0')

step_num(ax, 3.3, y, 1)
box(ax, cx, y, 7.0, 0.9,
    'CIFAR-10 Raw Dataset (.mat)  |  5 train batches + 1 test batch\n'
    '50,000 train + 10,000 test  |  10 classes  |  32×32×3 RGB, uint8',
    C['data'], bold=True, fontsize=9.5)

# ================================================================
# STAGE 2: IMAGE ENHANCEMENT
# ================================================================
y_enh = 15.0
section_bg(ax, 1, y_enh - 0.7, 14, 1.6, '#E8F5E9', '#A5D6A7')
ax.text(cx, y_enh + 0.6, 'Stage 2: Image Enhancement (per image)', fontsize=12,
        fontweight='bold', ha='center', color='#2E7D32')

arr_down(ax, cx, y - 0.45, y_enh + 0.45)

ew = 3.6
ex = [3.0, 8.0, 13.0]

step_num(ax, ex[0] - ew/2 + 0.1, y_enh + 0.45, 2)
box(ax, ex[0], y_enh, ew, 0.8,
    'Gaussian Denoising\nKernel 3×3, σ=0.5',
    C['denoise'], bold=True, fontsize=8.5, ec='#2E7D32')

arr_right(ax, ex[0] + ew/2, ex[1] - ew/2, y_enh, '#2E7D32')

step_num(ax, ex[1] - ew/2 + 0.1, y_enh + 0.45, 3)
box(ax, ex[1], y_enh, ew, 0.8,
    'CLAHE\nclipLimit=2.0, tile=4×4',
    C['clahe'], bold=True, fontsize=8.5, ec='#E65100')

arr_right(ax, ex[1] + ew/2, ex[2] - ew/2, y_enh, '#E65100')

step_num(ax, ex[2] - ew/2 + 0.1, y_enh + 0.45, 4)
box(ax, ex[2], y_enh, ew, 0.8,
    'Unsharp Masking\nWeight 1.5×, Blur −0.5×',
    C['sharpen'], bold=True, fontsize=8.5, ec='#C62828')

# ================================================================
# STAGE 3: DATA AUGMENTATION
# ================================================================
y_aug = 12.5
section_bg(ax, 1, y_aug - 1.8, 14, 2.8, '#F3E5F5', '#CE93D8')
ax.text(cx, y_aug + 0.7, 'Stage 3: Data Augmentation (50K → 75K)', fontsize=12,
        fontweight='bold', ha='center', color='#6A1B9A')

arr_down(ax, cx, y_enh - 0.4, y_aug + 0.45)

step_num(ax, 3.5, y_aug, 5)
box(ax, cx, y_aug, 7.0, 0.8,
    'Generate +25,000 augmented images (2,500 per class, balanced)\n'
    'Applied to preprocessed training images only',
    C['augment'], bold=True, fontsize=9, ec='#6A1B9A')

# Technique boxes
tech_y = y_aug - 1.2
techniques = ['Horizontal\nFlip (p=0.5)', 'Rotation\n±15°', 'Random Crop\n32×32 pad', 'Brightness\n×[0.8, 1.2]', 'Cutout\n10×10 (p=0.5)']
tw = 2.4
n = len(techniques)
total_w = n * tw + (n - 1) * 0.35
start_x = cx - total_w / 2 + tw / 2

for i, t in enumerate(techniques):
    tx = start_x + i * (tw + 0.35)
    box(ax, tx, tech_y, tw, 0.7, t, '#EDE7F6', fontsize=7.5, ec='#7B1FA2')
    arr_down(ax, tx, y_aug - 0.4, tech_y + 0.35, '#7B1FA2')

# ================================================================
# STAGE 4: NORMALIZATION
# ================================================================
y_norm = 9.8
section_bg(ax, 1, y_norm - 0.7, 14, 1.5, '#E0F7FA', '#80DEEA')
ax.text(cx, y_norm + 0.55, 'Stage 4: Normalization & Standardization', fontsize=12,
        fontweight='bold', ha='center', color='#00838F')

arr_down(ax, cx, tech_y - 0.35, y_norm + 0.4)

step_num(ax, 2.8, y_norm, 6)
box(ax, cx, y_norm, 9.0, 0.7,
    '÷ 255 → [0, 1]   then   x = (x − μ) / σ\n'
    'μ = [0.218, 0.218, 0.218]     σ = [0.350, 0.350, 0.346]',
    C['norm'], bold=True, fontsize=9.5, lw=1.5, ec='#00838F')

# ================================================================
# STAGE 5: SPLIT & SAVE
# ================================================================
y_split = 7.8
section_bg(ax, 1, y_split - 3.0, 14, 3.8, '#FFF9C4', '#FFF176')
ax.text(cx, y_split + 0.55, 'Stage 5: Train / Validation / Test Split', fontsize=12,
        fontweight='bold', ha='center', color='#F57F17')

arr_down(ax, cx, y_norm - 0.35, y_split + 0.4)

step_num(ax, 3.5, y_split, 7)
box(ax, cx, y_split, 7.0, 0.7,
    'Stratified Split (sklearn, random_state=42)\n'
    'Preserves class balance across subsets',
    C['split'], bold=True, fontsize=9, ec='#F57F17')

# Three output boxes
out_y = 6.3
out_w = 3.5
out_h = 0.9
out_x = [3.2, 8.0, 12.8]

arr_diag(ax, cx - 1.5, y_split - 0.35, out_x[0], out_y + out_h/2)
arr_down(ax, cx, y_split - 0.35, out_y + out_h/2)
arr_diag(ax, cx + 1.5, y_split - 0.35, out_x[2], out_y + out_h/2)

box(ax, out_x[0], out_y, out_w, out_h,
    'Training Set\n67,500 images (90%)',
    '#C8E6C9', bold=True, fontsize=9.5, lw=1.5, ec='#2E7D32')

box(ax, out_x[1], out_y, out_w, out_h,
    'Validation Set\n7,500 images (10%)',
    '#BBDEFB', bold=True, fontsize=9.5, lw=1.5, ec='#1565C0')

box(ax, out_x[2], out_y, out_w, out_h,
    'Test Set\n10,000 images',
    '#FFE0B2', bold=True, fontsize=9.5, lw=1.5, ec='#E65100')

# Save box
save_y = 5.1
arr_down(ax, out_x[0], out_y - out_h/2, save_y + 0.3)
arr_down(ax, out_x[1], out_y - out_h/2, save_y + 0.3)
arr_down(ax, out_x[2], out_y - out_h/2, save_y + 0.3)

step_num(ax, cx - 3.5, save_y, 8)
box(ax, cx, save_y, 6.5, 0.55,
    'Save → cifar10_preprocessed.npz  (381.8 MB)',
    C['save'], bold=True, fontsize=11, lw=2.0, ec='#C62828')

# ================================================================
# Legend
# ================================================================
legend_elements = [
    mpatches.Patch(facecolor=C['data'], edgecolor=C['border'], label='Data Loading'),
    mpatches.Patch(facecolor=C['denoise'], edgecolor='#2E7D32', label='Denoising'),
    mpatches.Patch(facecolor=C['clahe'], edgecolor='#E65100', label='CLAHE'),
    mpatches.Patch(facecolor=C['sharpen'], edgecolor='#C62828', label='Sharpening'),
    mpatches.Patch(facecolor=C['augment'], edgecolor='#6A1B9A', label='Augmentation'),
    mpatches.Patch(facecolor=C['norm'], edgecolor='#00838F', label='Normalization'),
    mpatches.Patch(facecolor=C['split'], edgecolor='#F57F17', label='Data Split'),
    mpatches.Patch(facecolor=C['save'], edgecolor='#C62828', label='Output'),
]
ax.legend(handles=legend_elements, loc='lower center', fontsize=8.5, ncol=4,
          framealpha=0.95, bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.savefig(os.path.join(FIG_DIR, 'preprocessing_pipeline.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Preprocessing pipeline diagram saved.")
