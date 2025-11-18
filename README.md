# Hybrid PointNet-Transformer-Fusion (PTF) for Tree Species Classification

This repository contains the official PyTorch implementation for the paper: **"Tree Species Classification using Hybrid PointNet-Transformer-Fusion and Structural Traits"**.

This project introduces the Hybrid PTF architecture, a novel deep learning model that classifies 3D LiDAR tree point clouds by fusing deep geometric features with domain-specific ecological knowledge (structural traits).

## 🏛️ Architecture

The PTF model uses a hybrid approach to capture three critical feature types:

1.  **PointNet Backbone**: Extracts a global feature vector encoding the overall tree shape.
2.  **Transformer Encoder**: Applies a self-attention mechanism to the point features to model long-range, non-local relationships across the tree crown.
3.  **Structural Trait Fusion**: Injects explicit ecological domain knowledge ($\mathbf{F}_{struct}$)—specifically **tree height ($h$)**, **crown diameter ($d_c$)**, and **canopy density ($\rho$)**.

These two feature streams ($\mathbf{F}_{deep}$ and $\mathbf{F}_{struct}$) are concatenated and passed through a final fusion layer for classification.

## 📈 Performance

The Hybrid PTF model achieved a **94.3% overall accuracy**, a **5.7 percentage point gain** over the PointNet++ baseline (88.6%).

* **Transformer Gain**: +3.4%
* **Structural Trait Gain**: +2.3%

## 🚀 Getting Started

### 1. Prerequisites

This project uses Python 3.8+ and PyTorch. All required libraries are listed in `requirements.txt`.

```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/Hybrid_PTF_Tree_Classification.git](https://github.com/YOUR_USERNAME/Hybrid_PTF_Tree_Classification.git)
cd Hybrid_PTF_Tree_Classification

# Install dependencies
pip install -r requirements.txt
```

### 2. Data

This model expects a dataset of individual tree point clouds, pre-segmented and normalized.

* **Format**: Each tree should be a `.txt` or `.npy` file containing $K$ points (e.g., $4096 \times 3$).
* **Preprocessing**: The paper uses Farthest Point Sampling (FPS) to downsample each tree to $K=4096$ points.
* **Location**: Place your training/testing data in the `/data/` directory.

### 3. Training

To train the model, run the `train.py` script:

```bash
python src/train.py
```

The script will load the data, build the Hybrid PTF model, and begin the training process.

### 4. Project Structure

```
Hybrid_PTF_Tree_Classification/
├── data/               # Dataset files
├── src/
│   ├── dataset.py      # Data loading logic
│   ├── model.py        # The Hybrid PTF model architecture
│   ├── utils.py        # Helper functions (trait calculation)
│   └── train.py        # Main training script
├── .gitignore          # Files to ignore
├── README.md           # This file
└── requirements.txt    # Python dependencies
```
