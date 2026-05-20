# FDL Project

CIFAR-10 preprocessing and CNN training notebooks for the Deep Learning course project.

## Project Overview

This repository contains notebook-based experiments for:

- CIFAR-10 preprocessing pipelines
- Data augmentation and normalization
- CNN and ResNet training
- Evaluation artifacts such as plots, confusion matrices, and saved weights

## Main Files

- `cifar10_preprocessing.ipynb` - original preprocessing workflow
- `cifar10_preprocessing_v2.ipynb` - preprocessing with denoising, CLAHE, sharpening, and augmentation
- `cifar10_preprocessing_v3.ipynb` - later preprocessing variant
- `cifar10_cnn_training.ipynb` - CNN training notebook
- `cifar10_resnet_training.ipynb` - ResNet training notebook
- `readme.html` - original CIFAR-10 dataset reference page

## Generated Artifacts

The repository may also include generated figures and model checkpoints such as:

- training curves
- confusion matrices
- sample predictions
- saved `.pth` model weights
- preprocessed `.npz` datasets

Large raw dataset and model artifacts are ignored by git when possible.

## Requirements

Typical Python packages used in the notebooks include:

- numpy
- scipy
- matplotlib
- pillow
- scikit-learn
- opencv-python
- torch

## How to Run

1. Open the notebook you want to execute in VS Code or Jupyter.
2. Make sure the CIFAR-10 files are available in the repository folder.
3. Run the cells in order from top to bottom.
4. If needed, install the notebook dependencies from the first cell.

## Notes

- The notebooks are designed to run on the CIFAR-10 batch files stored in this folder.
- Some notebooks save intermediate outputs such as `.png` figures and `.npz` files.
- The repository uses `main` as the default branch.
