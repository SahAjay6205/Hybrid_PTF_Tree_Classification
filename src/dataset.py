import os
import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import calculate_structural_traits, farthest_point_sample

class TreePCDataset(Dataset):
    """
    Custom PyTorch Dataset for loading tree point clouds and structural traits.
    """
    def __init__(self, data_root, split='train', num_points=4096, use_fps=True):
        """
        Args:
            data_root (str): Path to the main data folder.
            split (str): 'train', 'validation', or 'test'.
            num_points (int): Target number of points (K).
            use_fps (bool): Whether to use Farthest Point Sampling.
        """
        self.data_root = data_root
        self.num_points = num_points
        self.use_fps = use_fps
        self.split = split
        
        # --- This is just a placeholder ---
        # In a real project, you would load a file list from a .txt file
        # or scan a directory (e.g., /data/train_files.txt)
        # We will use the dummy data
        
        self.data_files = [
            ("dummy_trees/tree_01.txt", 0), # (path, class_label)
            ("dummy_trees/tree_02.txt", 1)  # (path, class_label)
        ]
        
        # In a real project, you would also pre-compute trait statistics
        # for normalization (mean and std for h, dc, p)
        self.trait_mean = np.array([10.0, 5.0, 50.0]) # Dummy mean
        self.trait_std = np.array([2.0, 1.0, 10.0])  # Dummy std

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        # 1. Load Data
        file_path, label = self.data_files[index]
        full_path = os.path.join(self.data_root, file_path)
        
        # Load point cloud (this is a dummy loader)
        # In a real project: point_cloud = np.loadtxt(full_path, delimiter=',')
        # We create fake data based on the index
        if index % 2 == 0:
            point_cloud = np.random.rand(5000, 3) * 10 # 5000 points
        else:
            point_cloud = np.random.rand(3000, 3) * 8  # 3000 points
        
        # 2. Preprocessing
        
        # Center the point cloud
        centroid = np.mean(point_cloud, axis=0)
        point_cloud = point_cloud - centroid
        
        # 3. Calculate Structural Traits (F_struct)
        # This MUST be done *before* downsampling to get accurate metrics
        traits_raw = calculate_structural_traits(point_cloud)
        
        # Normalize traits
        traits_normalized = (traits_raw - self.trait_mean) / self.trait_std
        traits_tensor = torch.tensor(traits_normalized, dtype=torch.float32)

        # 4. Downsample Point Cloud (F_deep input)
        # Convert to tensor for FPS
        point_cloud_tensor = torch.tensor(point_cloud, dtype=torch.float32)
        
        if self.use_fps:
            points = farthest_point_sample(point_cloud_tensor, self.num_points)
        else:
            # Simpler random sampling if FPS is not needed
            if len(point_cloud) >= self.num_points:
                indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            else:
                # Pad with duplicates if not enough points
                indices = np.random.choice(len(point_cloud), self.num_points, replace=True)
            points = point_cloud_tensor[indices, :]

        # 5. Data Augmentation (for training)
        if self.split == 'train':
            # Add augmentation like rotation, scaling, jittering
            pass 

        return points, traits_tensor, torch.tensor(label, dtype=torch.long)
