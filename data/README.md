This directory is intended for the tree point cloud dataset.

## Data Format

The model expects individual tree point clouds.
- Each file should represent one tree.
- The files should be text-based (e.g., .txt, .csv) or NumPy files (.npy).
- Each file should contain N x 3 coordinates (x, y, z).

## Preprocessing

As per the paper, each tree point cloud must be downsampled to a fixed size of K=4096 points using Farthest Point Sampling (FPS) before being fed into the model.

## Dummy Data

The `dummy_trees/` folder contains placeholder .txt files to demonstrate the expected structure. These are NOT real data and are just for testing the data loading pipeline.
