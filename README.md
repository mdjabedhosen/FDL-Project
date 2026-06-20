# CIFAR-10 Deep Learning — TensorFlow Implementation Documentation

This document explains every step and technique used in the TensorFlow implementation, and **why** each choice was made.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Preprocessing Notebook](#2-preprocessing-notebook)
3. [Custom CNN Notebook](#3-custom-cnn-notebook)
4. [Transfer Learning Notebook](#4-transfer-learning-notebook)
5. [Summary of Techniques](#5-summary-of-techniques)

---

## 1. Project Overview

**Goal:** Classify CIFAR-10 images (10 classes, 32×32 RGB) using three approaches:
- A custom CNN trained from scratch
- ResNet50V2 fine-tuned via transfer learning
- EfficientNetB0 fine-tuned via transfer learning

**Why three models?** To compare a small custom architecture against well-known pretrained backbones, and to show the practical benefit of transfer learning when the dataset is relatively small.

---

## 2. Preprocessing Notebook

File: `Preprocessing/cifar10_preprocessing.ipynb`

### Step 1 — Load Dataset
Read all 5 training batches + 1 test batch from `.mat` files, reshape from `(N, 3072)` → `(N, 32, 32, 3)`.

**Why:** CIFAR-10 is stored as flat byte arrays; we need image tensors for image processing and model input.

### Step 2 — Denoising (Gaussian Blur 3×3, σ=0.5)
Apply a small Gaussian kernel to each image.

**Why:** CIFAR-10 images are low-resolution and contain sensor/JPEG noise. A light blur removes high-frequency noise without destroying useful detail. Keeping σ small (0.5) preserves edges.

### Step 3 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
Applied per-channel with `clipLimit=2.0`, `tileGridSize=(4, 4)`.

**Why:** Many CIFAR images are dark or low-contrast. Standard histogram equalization can over-amplify noise; **CLAHE limits the contrast boost locally** so dark regions get enhanced without blowing out bright ones. Result: better class-discriminative features (edges, textures).

### Step 4 — Unsharp Masking (Sharpening)
`sharpened = 1.5 × image − 0.5 × blurred(image)`

**Why:** After denoising + CLAHE, edges become slightly soft. Unsharp masking restores edge crispness, which matters because convolutional filters depend heavily on edges.

### Step 5 — Data Augmentation (50K → 75K, +25K balanced)
Augmentations applied: **horizontal flip, rotation (±15°), random crop (pad-4 + crop), brightness (0.8–1.2), Cutout (10×10 patch)**.

**Why each augmentation:**
| Augmentation | Reason |
|---|---|
| Horizontal flip | Cars/animals look natural in both directions → free 2× data |
| Rotation ±15° | Real photos aren't perfectly aligned; teaches rotation invariance |
| Random crop (pad-reflect + crop) | Simulates translation; forces model not to rely on object position |
| Brightness jitter | Robustness to lighting conditions |
| **Cutout** | Forces the model to use *multiple* features instead of relying on one region — acts as strong regularization, reduces overfitting |

We generate exactly **2,500 augmented images per class** to keep the dataset **class-balanced**.

### Step 6 — Per-Channel Standardization
`X = (X − channel_mean) / channel_std`

**Why:** Neural networks train faster and more stably when inputs have zero mean and unit variance. Per-channel (not per-image) preserves color information.

### Step 7 — Train/Val Split (90/10 stratified)
`67.5K train / 7.5K val / 10K test`

**Why stratified:** Keeps the class distribution identical across splits — important for fair validation.

### Step 8 — Save as `.npz`
Compressed NumPy archive with all arrays + metadata (means, stds, label names).

**Why:** Run preprocessing **once** (it's slow). Training notebooks load instantly and reproducibly.

---

## 3. Custom CNN Notebook

File: `Model/cifar10_cnn_training.ipynb`

### Architecture (3 conv blocks → classifier)

```
Block 1: Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout(0.4)
Block 2: Conv(128) → BN → ReLU → Conv(128) → BN → ReLU → MaxPool → Dropout(0.5)
Block 3: Conv(256) → BN → ReLU → Conv(256) → BN → ReLU → MaxPool → Dropout(0.5)
Head:    Flatten → Dense(512) → BN → ReLU → Dropout(0.6) → Dense(10, softmax)
```

**Why this design:**

| Component | Why |
|---|---|
| **3 conv blocks (64→128→256)** | Standard "double the channels as you halve the resolution" pattern — keeps computation roughly constant per block while growing feature richness |
| **Two 3×3 convs per block** | Two stacked 3×3 convs = same receptive field as one 5×5 but fewer parameters and more non-linearity |
| **BatchNorm after every conv** | Stabilizes training, allows higher learning rates, acts as mild regularization |
| **ReLU** | Simple, fast, avoids vanishing gradients |
| **MaxPool 2×2** | Downsamples spatially, gives translation invariance |
| **Increasing Dropout (0.4→0.5→0.5→0.6)** | Deeper layers have more parameters → more overfitting risk → more dropout |
| **Dense(512) before output** | Mixes spatial features into class scores |
| **Softmax output** | Produces a valid probability distribution over 10 classes |

Total: ~3.25M parameters — small enough to train without a GPU cluster.

### Training Setup

| Setting | Value | Why |
|---|---|---|
| Loss | `SparseCategoricalCrossentropy` | Standard for multi-class with integer labels — no need to one-hot encode |
| Optimizer | `SGD(lr=0.01, momentum=0.9)` | SGD+momentum generalizes better than Adam on image tasks (well-documented finding) |
| LR Schedule | `CosineDecay` | Starts high (fast learning), smoothly decays to ~0 (fine adjustments at the end) |
| Batch Size | 128 | Sweet spot for stability vs. memory |
| Epochs | 100 (max) | Enough room for convergence |
| Early Stopping | `patience=10`, `monitor=val_accuracy` | Stops when val acc hasn't improved for 10 epochs — saves time, prevents overfitting |
| ModelCheckpoint | `save_best_only=True` | Always keep the best weights, not the last |

### Evaluation
- **Accuracy** (overall + per-class)
- **Classification Report** (precision, recall, F1 per class)
- **Confusion Matrix** (where the model gets confused — e.g. cat↔dog)
- **Sample Predictions** (visual check of right/wrong cases)

**Why all four:** Accuracy alone is misleading. Per-class metrics show **which** classes are hard. Confusion matrix reveals **which pairs** are confused. Visual samples build intuition.

---

## 4. Transfer Learning Notebook

File: `Model/cifar10_transfer_learning.ipynb`

### Why Transfer Learning?

Training a deep model (millions of params) on only ~75K small images is risky — likely to overfit. **Pretrained models** have already learned generic visual features (edges, textures, shapes) from ImageNet (1.2M images, 1000 classes). We reuse those features and only re-learn the final classifier for CIFAR-10.

### Models Used

| Model | Params | Why chosen |
|---|---|---|
| **ResNet50V2** | ~25M | Industry-standard backbone, residual connections prevent vanishing gradients in deep nets. (Keras has no ResNet18, so we use the closest standard alternative.) |
| **EfficientNetB0** | ~5M | State-of-the-art accuracy-per-parameter ratio thanks to **compound scaling** (depth + width + resolution balanced together) |

### Two-Phase Strategy

**Phase A — Train head only (10 epochs)**
1. Load pretrained backbone with `weights='imagenet'`
2. `base_model.trainable = False` (freeze)
3. Replace top: `GlobalAveragePooling2D → Dropout(0.3) → Dense(10, softmax)`
4. Train with **Adam(lr=0.001)** — only the new head's ~10K params update

**Why:** If we immediately unfreeze and use a normal learning rate, the random new head produces large gradients that would **destroy** the pretrained features. We first let the head "catch up" to a reasonable state.

**Phase B — Fine-tune everything (20 epochs)**
1. `base_model.trainable = True` (unfreeze)
2. Switch to **SGD(lr=0.001, momentum=0.9)** — very small LR
3. Train with `EarlyStopping(patience=5)` + `ModelCheckpoint`

**Why small LR:** We don't want big updates to pretrained weights — just gentle adjustments to specialize them for CIFAR-10. SGD again because it generalizes better than Adam for fine-tuning.

### Input Pipeline (`tf.data`)

We use `tf.data.Dataset` with:
- `.map(preprocess, num_parallel_calls=AUTOTUNE)` — resize 32×32→224×224 + model-specific normalization, in parallel
- `.shuffle(10000)` (train only) — break order correlation
- `.batch(64)` — small batch since 224×224 is memory-heavy
- `.prefetch(AUTOTUNE)` — overlap data prep with GPU compute

**Why on-the-fly resize:** Storing 75K × 224×224×3 floats = ~45 GB. Doing it in the data pipeline keeps memory tiny.

**Why model-specific preprocessing:**
- `keras.applications.resnet_v2.preprocess_input` — scales pixels to `[-1, 1]`
- `keras.applications.efficientnet.preprocess_input` — uses the exact stats the model was pretrained with

If we use the wrong preprocessing, the pretrained features won't activate correctly → much worse accuracy.

### Why GlobalAveragePooling instead of Flatten?

`GlobalAveragePooling2D` averages each feature map to a single number. Compared to `Flatten` + huge Dense layer:
- Far fewer parameters (no overfitting)
- Spatially invariant (good for classification)
- Standard practice for modern CNNs

---

## 5. Summary of Techniques

| Technique | Where Used | Why It Helps |
|---|---|---|
| Gaussian denoising | Preprocessing | Removes sensor noise |
| CLAHE | Preprocessing | Boosts local contrast safely |
| Unsharp masking | Preprocessing | Restores edge sharpness |
| Horizontal flip | Augmentation | Free 2× data |
| Rotation/Crop/Brightness | Augmentation | Generalization to unseen variations |
| **Cutout** | Augmentation | Forces use of multiple features, strong regularizer |
| Per-channel standardization | Preprocessing | Stable, fast training |
| Stratified split | Train/val | Fair class distribution |
| BatchNorm | CNN | Stable, fast training |
| Dropout (increasing) | CNN | Prevents overfitting in deep layers |
| SGD + Momentum | CNN + Fine-tune | Better generalization than Adam for vision |
| Cosine LR Decay | CNN | Smooth learning rate annealing |
| Early Stopping | All training | Saves time, prevents overfitting |
| ModelCheckpoint | All training | Always keep best weights |
| **Transfer Learning** | ResNet/EffNet | Reuse ImageNet knowledge |
| **Freeze → Unfreeze** | Transfer | Protects pretrained weights |
| GlobalAveragePooling | Transfer head | Few params, spatial invariance |
| `tf.data` pipeline | Transfer | Efficient on-the-fly resize |
| Model-specific `preprocess_input` | Transfer | Matches pretraining normalization |

---

## How to Run

1. Run `Preprocessing/cifar10_preprocessing.ipynb` once → produces `cifar10_preprocessed.npz`
2. Run `Model/cifar10_cnn_training.ipynb` → trains custom CNN, saves `best_cnn_model.h5`
3. Run `Model/cifar10_transfer_learning.ipynb` → trains ResNet50V2 + EfficientNetB0, saves `.h5` files
4. All figures land in `Figure/` for inclusion in the final report.
