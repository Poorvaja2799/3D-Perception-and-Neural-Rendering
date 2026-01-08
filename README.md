# 3D Perception and Neural Rendering

This repository contains two self-contained projects for 3D understanding and rendering:

- Point cloud classification with PointNet and variants (Classifying Point Clouds with PointNet)
- Neural Radiance Fields (NeRF) rendering pipeline (NeRF)

Each project has its own Conda environment and Python package (both named `vision`) scoped to that environment. Use separate environments to avoid conflicts.

## Repository Structure
- [Classifying Point Clouds with PointNet](Classifying%20Point%20Clouds%20with%20PointNet): PointNet models, dataloaders, training utilities, and a notebook.
- [NeRF](NeRF): NeRF components (positional encoding, ray sampling, compositing), and a notebook.
- [README.md](README.md): This document.

## Prerequisites
- macOS with Conda (miniconda/anaconda).
- Python 3.10 (managed by the provided `environment.yml`).
- PyTorch and TorchVision (installed via Conda).
- Optional GPU acceleration:
	- NVIDIA CUDA on Linux/Windows (`device='cuda'`).
	- Apple Silicon MPS on macOS (`device='mps'` with compatible PyTorch).

## Setup

Create separate Conda environments and install each project in editable mode.

### PointNet Environment
From the folder [Classifying Point Clouds with PointNet](Classifying%20Point%20Clouds%20with%20PointNet):

```bash
conda env create -f conda/environment.yml
conda activate cv_proj5
pip install -e .
```

### NeRF Environment
From the folder [NeRF](NeRF):

```bash
conda env create -f conda/environment.yml
conda activate cv_proj6
pip install -e .
```

## Data

PointNet data is organized under [Classifying Point Clouds with PointNet/data/sweeps](Classifying%20Point%20Clouds%20with%20PointNet/data/sweeps), with one folder per class containing 200 `.txt` point cloud files (`0.txt` to `199.txt`). The `Argoverse` dataset splits are:
- Train: indices 0–169
- Test: indices 170–199

## Quick Start

### Run the Notebooks

Jupyter notebooks provide end-to-end workflows. Launch Jupyter in the corresponding environment:

```bash
# PointNet notebook
conda activate cv_proj5
jupyter notebook Classifying\ Point\ Clouds\ with\ PointNet/proj5.ipynb

# NeRF notebook
conda activate cv_proj6
jupyter notebook NeRF/proj6_local.ipynb
```

### PointNet: Train and Evaluate (Python)

Use the packaged modules inside [Classifying Point Clouds with PointNet/src/vision](Classifying%20Point%20Clouds%20with%20PointNet/src/vision):

Models are saved automatically to [Classifying Point Clouds with PointNet/output](Classifying%20Point%20Clouds%20with%20PointNet/output) with filenames matching the class name, e.g., `Baseline.pt`, `PointNet.pt`, `PointNetTNet.pt`.

### NeRF: Render an Image (Python)

Use components under [NeRF/src/vision](NeRF/src/vision):

The NeRF notebook demonstrates loading the provided sample dataset [NeRF/lego_data_update.npz](NeRF/lego_data_update.npz) and training/rendering end-to-end. Trained weights, if saved, are expected under [NeRF/output](NeRF/output).
