import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

COLORS = {
    'input': '#E8F5E9',
    'conv': '#BBDEFB',
    'bn_relu': '#E3F2FD',
    'pool': '#C8E6C9',
    'dropout': '#FFF9C4',
    'dense': '#FFE0B2',
    'output': '#F8BBD0',
    'pretrained': '#E1BEE7',
    'resize': '#B2DFDB',
    'border': '#37474F',
    'arrow': '#546E7A',
}

def draw_block(ax, x, y, w, h, text, color, fontsize=8, bold=False, border_width=1.0):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=COLORS['border'],
                         linewidth=border_width, zorder=2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, zorder=3, wrap=True)
    return y

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                                lw=1.2, connectionstyle='arc3,rad=0'))

def draw_dim_label(ax, x, y, text):
    ax.text(x, y, text, ha='center', va='center', fontsize=6.5,
            color='#616161', style='italic', zorder=3)


# ============================================================
# DIAGRAM 1: Custom CNN
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 7))
ax.set_xlim(-1, 15)
ax.set_ylim(-0.5, 7)
ax.axis('off')
ax.set_title('(a) Custom CNN Architecture', fontsize=13, fontweight='bold', pad=15)

bw = 1.6
bh = 0.55
gap = 0.75

# Row positions
row1_y = 6.0
row2_y = 4.5
row3_y = 3.0
row4_y = 1.5

# --- Input ---
x = 0.8
draw_block(ax, x, row1_y, bw, bh, 'Input\n32×32×3', COLORS['input'], bold=True)

# --- Block 1 ---
x1 = 3.0
draw_arrow(ax, 0.8 + bw/2, row1_y, x1 - bw/2, row1_y)
draw_block(ax, x1, row1_y, bw, bh, 'Conv2D(64)\n3×3, same', COLORS['conv'])
x2 = x1 + bw + 0.3
draw_arrow(ax, x1 + bw/2, row1_y, x2 - bw/2, row1_y)
draw_block(ax, x2, row1_y, bw*0.7, bh, 'BN→ReLU', COLORS['bn_relu'], fontsize=7)
x3 = x2 + bw*0.7/2 + 0.3 + bw/2
draw_arrow(ax, x2 + bw*0.7/2, row1_y, x3 - bw/2, row1_y)
draw_block(ax, x3, row1_y, bw, bh, 'Conv2D(64)\n3×3, same', COLORS['conv'])
x4 = x3 + bw/2 + 0.3 + bw*0.7/2
draw_arrow(ax, x3 + bw/2, row1_y, x4 - bw*0.7/2, row1_y)
draw_block(ax, x4, row1_y, bw*0.7, bh, 'BN→ReLU', COLORS['bn_relu'], fontsize=7)
x5 = x4 + bw*0.7/2 + 0.3 + bw*0.6/2
draw_arrow(ax, x4 + bw*0.7/2, row1_y, x5 - bw*0.6/2, row1_y)
draw_block(ax, x5, row1_y, bw*0.6, bh, 'MaxPool\n2×2', COLORS['pool'], fontsize=7)
x6 = x5 + bw*0.6/2 + 0.3 + bw*0.6/2
draw_arrow(ax, x5 + bw*0.6/2, row1_y, x6 - bw*0.6/2, row1_y)
draw_block(ax, x6, row1_y, bw*0.6, bh, 'Drop\n0.4', COLORS['dropout'], fontsize=7)
draw_dim_label(ax, x6 + bw*0.6/2 + 0.5, row1_y, '16×16×64')

# Block label
ax.text(7, row1_y + 0.5, 'Block 1', fontsize=9, fontweight='bold', color='#1565C0', ha='center')

# --- Block 2 ---
x = 0.8
draw_arrow(ax, x6 + bw*0.6/2, row1_y - bh/2, x + bw/2, row2_y + bh/2)
draw_block(ax, 3.0, row2_y, bw, bh, 'Conv2D(128)\n3×3, same', COLORS['conv'])
draw_arrow(ax, 0.8 + bw/2, row2_y, 3.0 - bw/2, row2_y)
draw_block(ax, 0.8, row2_y, bw*0.01, bh*0.01, '', 'white')  # invisible anchor

x1 = 3.0
x2 = x1 + bw + 0.3
draw_arrow(ax, x1 + bw/2, row2_y, x2 - bw/2, row2_y)
draw_block(ax, x2, row2_y, bw*0.7, bh, 'BN→ReLU', COLORS['bn_relu'], fontsize=7)
x3 = x2 + bw*0.7/2 + 0.3 + bw/2
draw_arrow(ax, x2 + bw*0.7/2, row2_y, x3 - bw/2, row2_y)
draw_block(ax, x3, row2_y, bw, bh, 'Conv2D(128)\n3×3, same', COLORS['conv'])
x4 = x3 + bw/2 + 0.3 + bw*0.7/2
draw_arrow(ax, x3 + bw/2, row2_y, x4 - bw*0.7/2, row2_y)
draw_block(ax, x4, row2_y, bw*0.7, bh, 'BN→ReLU', COLORS['bn_relu'], fontsize=7)
x5 = x4 + bw*0.7/2 + 0.3 + bw*0.6/2
draw_arrow(ax, x4 + bw*0.7/2, row2_y, x5 - bw*0.6/2, row2_y)
draw_block(ax, x5, row2_y, bw*0.6, bh, 'MaxPool\n2×2', COLORS['pool'], fontsize=7)
x6 = x5 + bw*0.6/2 + 0.3 + bw*0.6/2
draw_arrow(ax, x5 + bw*0.6/2, row2_y, x6 - bw*0.6/2, row2_y)
draw_block(ax, x6, row2_y, bw*0.6, bh, 'Drop\n0.5', COLORS['dropout'], fontsize=7)
draw_dim_label(ax, x6 + bw*0.6/2 + 0.5, row2_y, '8×8×128')
ax.text(7, row2_y + 0.5, 'Block 2', fontsize=9, fontweight='bold', color='#1565C0', ha='center')

# --- Block 3 ---
draw_block(ax, 3.0, row3_y, bw, bh, 'Conv2D(256)\n3×3, same', COLORS['conv'])
x1 = 3.0
x2 = x1 + bw + 0.3
draw_arrow(ax, x1 + bw/2, row3_y, x2 - bw/2, row3_y)
draw_block(ax, x2, row3_y, bw*0.7, bh, 'BN→ReLU', COLORS['bn_relu'], fontsize=7)
x3 = x2 + bw*0.7/2 + 0.3 + bw/2
draw_arrow(ax, x2 + bw*0.7/2, row3_y, x3 - bw/2, row3_y)
draw_block(ax, x3, row3_y, bw, bh, 'Conv2D(256)\n3×3, same', COLORS['conv'])
x4 = x3 + bw/2 + 0.3 + bw*0.7/2
draw_arrow(ax, x3 + bw/2, row3_y, x4 - bw*0.7/2, row3_y)
draw_block(ax, x4, row3_y, bw*0.7, bh, 'BN→ReLU', COLORS['bn_relu'], fontsize=7)
x5 = x4 + bw*0.7/2 + 0.3 + bw*0.6/2
draw_arrow(ax, x4 + bw*0.7/2, row3_y, x5 - bw*0.6/2, row3_y)
draw_block(ax, x5, row3_y, bw*0.6, bh, 'MaxPool\n2×2', COLORS['pool'], fontsize=7)
x6_b3 = x5 + bw*0.6/2 + 0.3 + bw*0.6/2
draw_arrow(ax, x5 + bw*0.6/2, row3_y, x6_b3 - bw*0.6/2, row3_y)
draw_block(ax, x6_b3, row3_y, bw*0.6, bh, 'Drop\n0.5', COLORS['dropout'], fontsize=7)
draw_dim_label(ax, x6_b3 + bw*0.6/2 + 0.5, row3_y, '4×4×256')
ax.text(7, row3_y + 0.5, 'Block 3', fontsize=9, fontweight='bold', color='#1565C0', ha='center')

# Connection block2 -> block3
draw_arrow(ax, x6 + bw*0.6/2, row2_y - bh/2, 3.0 - bw/2, row3_y + bh/2)

# --- Classifier ---
cx = 2.0
draw_arrow(ax, x6_b3 + bw*0.6/2, row3_y - bh/2, cx - bw*0.6/2, row4_y + bh/2)
draw_block(ax, cx, row4_y, bw*0.7, bh, 'Flatten', COLORS['pool'], fontsize=7)
draw_dim_label(ax, cx, row4_y - 0.45, '4096')

cx2 = cx + bw*0.7/2 + 0.4 + bw/2
draw_arrow(ax, cx + bw*0.7/2, row4_y, cx2 - bw/2, row4_y)
draw_block(ax, cx2, row4_y, bw, bh, 'Dense(512)', COLORS['dense'])

cx3 = cx2 + bw/2 + 0.3 + bw*0.7/2
draw_arrow(ax, cx2 + bw/2, row4_y, cx3 - bw*0.7/2, row4_y)
draw_block(ax, cx3, row4_y, bw*0.7, bh, 'BN→ReLU', COLORS['bn_relu'], fontsize=7)

cx4 = cx3 + bw*0.7/2 + 0.3 + bw*0.6/2
draw_arrow(ax, cx3 + bw*0.7/2, row4_y, cx4 - bw*0.6/2, row4_y)
draw_block(ax, cx4, row4_y, bw*0.6, bh, 'Drop\n0.6', COLORS['dropout'], fontsize=7)

cx5 = cx4 + bw*0.6/2 + 0.4 + bw/2
draw_arrow(ax, cx4 + bw*0.6/2, row4_y, cx5 - bw/2, row4_y)
draw_block(ax, cx5, row4_y, bw, bh, 'Dense(10)\nSoftmax', COLORS['output'], bold=True)

ax.text(7, row4_y + 0.5, 'Classifier', fontsize=9, fontweight='bold', color='#C62828', ha='center')
ax.text(7, 0.3, 'Total Parameters: 3.25M', fontsize=10, ha='center', fontweight='bold', color='#37474F')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['conv'], edgecolor=COLORS['border'], label='Convolution'),
    mpatches.Patch(facecolor=COLORS['bn_relu'], edgecolor=COLORS['border'], label='BatchNorm + ReLU'),
    mpatches.Patch(facecolor=COLORS['pool'], edgecolor=COLORS['border'], label='Pooling / Flatten'),
    mpatches.Patch(facecolor=COLORS['dropout'], edgecolor=COLORS['border'], label='Dropout'),
    mpatches.Patch(facecolor=COLORS['dense'], edgecolor=COLORS['border'], label='Dense'),
    mpatches.Patch(facecolor=COLORS['output'], edgecolor=COLORS['border'], label='Output'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=3, framealpha=0.9)

plt.tight_layout()
plt.savefig('/Users/mjabed3834/Documents/UNIMIB/Second Semester/Deep Learning/cifar-10-batches-mat/TensorFlow/Figure/arch_custom_cnn.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# ============================================================
# DIAGRAM 2: ResNet50V2 Transfer Learning
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(0, 5.5)
ax.axis('off')
ax.set_title('(b) ResNet50V2 Transfer Learning Architecture', fontsize=13, fontweight='bold', pad=15)

cy = 3.5
bw2 = 1.8

# Input
x = 0.9
draw_block(ax, x, cy, bw2, 0.7, 'Input\n32×32×3', COLORS['input'], bold=True)

# Resize
x2 = x + bw2/2 + 0.4 + bw2*0.7/2
draw_arrow(ax, x + bw2/2, cy, x2 - bw2*0.7/2, cy)
draw_block(ax, x2, cy, bw2*0.7, 0.7, 'Resize\n224×224', COLORS['resize'], fontsize=7)

# Preprocess
x3 = x2 + bw2*0.7/2 + 0.4 + bw2*0.8/2
draw_arrow(ax, x2 + bw2*0.7/2, cy, x3 - bw2*0.8/2, cy)
draw_block(ax, x3, cy, bw2*0.8, 0.7, 'Preprocess\nInput', COLORS['resize'], fontsize=7)

# ResNet backbone (large block)
backbone_w = 3.5
backbone_h = 1.8
x4 = x3 + bw2*0.8/2 + 0.5 + backbone_w/2
draw_arrow(ax, x3 + bw2*0.8/2, cy, x4 - backbone_w/2, cy)
box = FancyBboxPatch((x4 - backbone_w/2, cy - backbone_h/2), backbone_w, backbone_h,
                     boxstyle="round,pad=0.05",
                     facecolor=COLORS['pretrained'], edgecolor='#6A1B9A',
                     linewidth=2.0, zorder=2, linestyle='-')
ax.add_patch(box)
ax.text(x4, cy + 0.35, 'ResNet50V2', ha='center', va='center', fontsize=11,
        fontweight='bold', color='#4A148C', zorder=3)
ax.text(x4, cy - 0.05, 'ImageNet Pretrained', ha='center', va='center', fontsize=8,
        color='#6A1B9A', zorder=3, style='italic')
ax.text(x4, cy - 0.45, '23.6M params', ha='center', va='center', fontsize=7.5,
        color='#7B1FA2', zorder=3)

# GAP
x5 = x4 + backbone_w/2 + 0.4 + bw2*0.7/2
draw_arrow(ax, x4 + backbone_w/2, cy, x5 - bw2*0.7/2, cy)
draw_block(ax, x5, cy, bw2*0.7, 0.7, 'Global\nAvgPool', COLORS['pool'], fontsize=7)
draw_dim_label(ax, x5, cy - 0.55, '2048')

# Dropout
x6 = x5 + bw2*0.7/2 + 0.3 + bw2*0.5/2
draw_arrow(ax, x5 + bw2*0.7/2, cy, x6 - bw2*0.5/2, cy)
draw_block(ax, x6, cy, bw2*0.5, 0.7, 'Drop\n0.3', COLORS['dropout'], fontsize=7)

# Output
x7 = x6 + bw2*0.5/2 + 0.4 + bw2*0.8/2
draw_arrow(ax, x6 + bw2*0.5/2, cy, x7 - bw2*0.8/2, cy)
draw_block(ax, x7, cy, bw2*0.8, 0.7, 'Dense(10)\nSoftmax', COLORS['output'], bold=True)

# Training strategy
strat_y = 1.2
ax.text(7, strat_y + 0.6, 'Training Strategy', fontsize=10, fontweight='bold', ha='center', color='#37474F')
phase1 = 'Phase 1: Freeze backbone → Train head only (Adam, lr=0.001, 10 epochs)'
phase2 = 'Phase 2: Unfreeze all → Fine-tune (SGD, lr=0.001, momentum=0.9, 20 epochs)'
ax.text(7, strat_y, phase1, fontsize=8, ha='center', color='#455A64')
ax.text(7, strat_y - 0.4, phase2, fontsize=8, ha='center', color='#455A64')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['input'], edgecolor=COLORS['border'], label='Input'),
    mpatches.Patch(facecolor=COLORS['resize'], edgecolor=COLORS['border'], label='Resize / Preprocess'),
    mpatches.Patch(facecolor=COLORS['pretrained'], edgecolor='#6A1B9A', label='Pretrained Backbone'),
    mpatches.Patch(facecolor=COLORS['pool'], edgecolor=COLORS['border'], label='Global Avg Pool'),
    mpatches.Patch(facecolor=COLORS['dropout'], edgecolor=COLORS['border'], label='Dropout'),
    mpatches.Patch(facecolor=COLORS['output'], edgecolor=COLORS['border'], label='Output'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=3, framealpha=0.9)

plt.tight_layout()
plt.savefig('/Users/mjabed3834/Documents/UNIMIB/Second Semester/Deep Learning/cifar-10-batches-mat/TensorFlow/Figure/arch_resnet50v2.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()


# ============================================================
# DIAGRAM 3: EfficientNetB0 Transfer Learning
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(14, 5))
ax.set_xlim(-0.5, 14.5)
ax.set_ylim(0, 5.5)
ax.axis('off')
ax.set_title('(c) EfficientNetB0 Transfer Learning Architecture', fontsize=13, fontweight='bold', pad=15)

cy = 3.5

# Input
x = 0.9
draw_block(ax, x, cy, bw2, 0.7, 'Input\n32×32×3', COLORS['input'], bold=True)

# Resize
x2 = x + bw2/2 + 0.4 + bw2*0.7/2
draw_arrow(ax, x + bw2/2, cy, x2 - bw2*0.7/2, cy)
draw_block(ax, x2, cy, bw2*0.7, 0.7, 'Resize\n224×224', COLORS['resize'], fontsize=7)

# Preprocess
x3 = x2 + bw2*0.7/2 + 0.4 + bw2*0.8/2
draw_arrow(ax, x2 + bw2*0.7/2, cy, x3 - bw2*0.8/2, cy)
draw_block(ax, x3, cy, bw2*0.8, 0.7, 'Preprocess\nInput', COLORS['resize'], fontsize=7)

# EfficientNet backbone (large block)
backbone_w = 3.5
backbone_h = 1.8
x4 = x3 + bw2*0.8/2 + 0.5 + backbone_w/2
draw_arrow(ax, x3 + bw2*0.8/2, cy, x4 - backbone_w/2, cy)
box = FancyBboxPatch((x4 - backbone_w/2, cy - backbone_h/2), backbone_w, backbone_h,
                     boxstyle="round,pad=0.05",
                     facecolor=COLORS['pretrained'], edgecolor='#6A1B9A',
                     linewidth=2.0, zorder=2, linestyle='-')
ax.add_patch(box)
ax.text(x4, cy + 0.45, 'EfficientNetB0', ha='center', va='center', fontsize=11,
        fontweight='bold', color='#4A148C', zorder=3)
ax.text(x4, cy + 0.05, 'ImageNet Pretrained', ha='center', va='center', fontsize=8,
        color='#6A1B9A', zorder=3, style='italic')
ax.text(x4, cy - 0.3, '4.05M params', ha='center', va='center', fontsize=7.5,
        color='#7B1FA2', zorder=3)
ax.text(x4, cy - 0.6, 'Compound Scaling', ha='center', va='center', fontsize=7,
        color='#9C27B0', zorder=3, style='italic')
ax.text(x4, cy - 0.85, '(depth + width + resolution)', ha='center', va='center', fontsize=6.5,
        color='#AB47BC', zorder=3)

# GAP
x5 = x4 + backbone_w/2 + 0.4 + bw2*0.7/2
draw_arrow(ax, x4 + backbone_w/2, cy, x5 - bw2*0.7/2, cy)
draw_block(ax, x5, cy, bw2*0.7, 0.7, 'Global\nAvgPool', COLORS['pool'], fontsize=7)
draw_dim_label(ax, x5, cy - 0.55, '1280')

# Dropout
x6 = x5 + bw2*0.7/2 + 0.3 + bw2*0.5/2
draw_arrow(ax, x5 + bw2*0.7/2, cy, x6 - bw2*0.5/2, cy)
draw_block(ax, x6, cy, bw2*0.5, 0.7, 'Drop\n0.3', COLORS['dropout'], fontsize=7)

# Output
x7 = x6 + bw2*0.5/2 + 0.4 + bw2*0.8/2
draw_arrow(ax, x6 + bw2*0.5/2, cy, x7 - bw2*0.8/2, cy)
draw_block(ax, x7, cy, bw2*0.8, 0.7, 'Dense(10)\nSoftmax', COLORS['output'], bold=True)

# Training strategy
strat_y = 1.2
ax.text(7, strat_y + 0.6, 'Training Strategy', fontsize=10, fontweight='bold', ha='center', color='#37474F')
phase1 = 'Phase 1: Freeze backbone → Train head only (Adam, lr=0.001, 10 epochs)'
phase2 = 'Phase 2: Unfreeze all → Fine-tune (SGD, lr=0.001, momentum=0.9, 20 epochs)'
ax.text(7, strat_y, phase1, fontsize=8, ha='center', color='#455A64')
ax.text(7, strat_y - 0.4, phase2, fontsize=8, ha='center', color='#455A64')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['input'], edgecolor=COLORS['border'], label='Input'),
    mpatches.Patch(facecolor=COLORS['resize'], edgecolor=COLORS['border'], label='Resize / Preprocess'),
    mpatches.Patch(facecolor=COLORS['pretrained'], edgecolor='#6A1B9A', label='Pretrained Backbone'),
    mpatches.Patch(facecolor=COLORS['pool'], edgecolor=COLORS['border'], label='Global Avg Pool'),
    mpatches.Patch(facecolor=COLORS['dropout'], edgecolor=COLORS['border'], label='Dropout'),
    mpatches.Patch(facecolor=COLORS['output'], edgecolor=COLORS['border'], label='Output'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=3, framealpha=0.9)

plt.tight_layout()
plt.savefig('/Users/mjabed3834/Documents/UNIMIB/Second Semester/Deep Learning/cifar-10-batches-mat/TensorFlow/Figure/arch_efficientnetb0.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("All 3 architecture diagrams saved to TensorFlow/Figure/")
print("  - arch_custom_cnn.png")
print("  - arch_resnet50v2.png")
print("  - arch_efficientnetb0.png")
