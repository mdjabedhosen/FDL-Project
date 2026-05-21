# FDL Project

CIFAR-10 preprocessing, training, and evaluation notebooks for the Deep Learning course project.

## Repository Structure

```text
.
├── Dataset/                 # CIFAR-10 source files and reference page
├── Figure/                  # Saved plots and evaluation images
├── Model/                   # Model training notebooks
├── Preprocessing/           # Preprocessing notebooks and generated datasets
├── README.md
├── best_model.pth
└── best_model_v3.pth
```

## Notebooks

- `Preprocessing/cifar10_preprocessing.ipynb` - main preprocessing pipeline: load dataset, upscale, denoise, apply CLAHE, sharpen, augment, normalize, and export for ResNet training notebook 
- `Preprocessing/cifar10_preprocessing_v2.ipynb` - earlier preprocessing version
- `Preprocessing/cifar10_preprocessing_v3.ipynb` - later preprocessing version with out upscale for CNN training notebook and  transfer learning notebook
- `Model/cifar10_cnn_training.ipynb` - CNN training notebook
- `Model/cifar10_resnet_training.ipynb` - ResNet training notebook
- `Model/cifar10_transfer_learning.ipynb` - transfer learning notebook

## Data And Outputs

- `Dataset/` contains the CIFAR-10 `.mat` files and `readme.html`
- `Figure/` contains saved plots such as confusion matrices, training curves, sample predictions, and comparison figures
- `Preprocessing/` contains generated `.npz` files and preprocessing comparison images
- `best_model.pth` and `best_model_v3.pth` are saved model weights

## Requirements

The notebooks typically use:

- numpy
- scipy
- matplotlib
- pillow
- scikit-learn
- opencv-python
- torch

## How To Run

1. Open the notebook you want to run in VS Code or Jupyter.
2. Run the cells from top to bottom.
3. The preprocessing notebook now finds the dataset automatically from the repository tree.
4. Install any missing packages from the first code cell if needed.

## Notes

- Generated `.npz`, `.pth`, and CIFAR-10 `.mat` files are ignored by git.
- Some notebooks save images into `Figure/` or `Preprocessing/`.
- The default branch is `main`.
