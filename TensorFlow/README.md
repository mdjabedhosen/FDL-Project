# FDL-Project — CIFAR-10 TensorFlow

This folder contains TensorFlow-based code and notebooks used for experiments on the CIFAR-10 dataset.

Contents
- `Dataset/` — original CIFAR-10 batch .mat files used for preprocessing.
- `Preprocessing/` — notebook and scripts to convert and preprocess CIFAR-10 (`cifar10_preprocessing.ipynb`, `cifar10_preprocessed.npz`).
- `Model/` — training and transfer-learning notebooks and saved models (`*.h5`).
- `Figure/` — figures and diagrams produced by the notebooks.

Quick start
1. Create a Python virtual environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter and open the notebooks in `Model/` or `Preprocessing/`:

```bash
jupyter notebook
```

Notes
- The `requirements.txt` file was generated from imports used in the notebooks; adjust versions as needed for your environment.
- Large model files (`*.h5`) are present under `Model/`. If you re-train, saving weights may produce large files — ensure you have enough disk space.
- Notebook cells include optional `google.colab` sections for running in Colab; those will attempt to mount Google Drive when present.

Contributing
- Make changes on a feature branch and open a pull request to `clean-main`.

License
- Please verify and add a license at the repository root if needed.
