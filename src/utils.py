import torch
import numpy as np

def farthest_point_sample(point_cloud, num_points):
    """
    Implements Farthest Point Sampling (FPS) to downsample a point cloud.
    
    
    Args:
        point_cloud (torch.Tensor): The input point cloud (N, 3).
        num_points (int): The target number of points (K).
        
    Returns:
        torch.Tensor: The downsampled point cloud (K, 3).
    """
    N, D = point_cloud.shape
    if N == 0:
        return torch.zeros((num_points, 3), device=point_cloud.device)
        
    centroids = torch.zeros(num_points, dtype=torch.long, device=point_cloud.device)
    distance = torch.ones(N, device=point_cloud.device) * 1e10
    
    # Select a random point as the first centroid
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device=point_cloud.device)
    
    for i in range(num_points):
        centroids[i] = farthest
        centroid_point = point_cloud[farthest, :]
        
        # Calculate distance from the new centroid to all other points
        dist = torch.sum((point_cloud - centroid_point) ** 2, dim=-1)
        
        # Update distances: find the minimum distance from any centroid
        mask = dist < distance
        distance[mask] = dist[mask]
        
        # Select the point that is farthest from all centroids
        farthest = torch.max(distance, dim=-1)[1]
        
    return point_cloud[centroids]

def calculate_structural_traits(point_cloud_np):
    """
    Calculates the normalized structural traits F_struct [h, dc, p].
    
    
    Args:
        point_cloud_np (np.array): A NumPy array of the point cloud (N, 3).
        
    Returns:
        np.array: A 1D array (vector) of 3 structural traits.
    """
    if point_cloud_np.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 1. Tree Height (h)
    # Assumes point cloud is normalized, so min_z is ground (or close to 0)
    max_z = np.max(point_cloud_np[:, 2])
    min_z = np.min(point_cloud_np[:, 2])
    tree_height = max_z - min_z
    
    # 2. Crown Diameter (dc)
    # Average extent of the crown in x and y dimensions
    max_x = np.max(point_cloud_np[:, 0])
    min_x = np.min(point_cloud_np[:, 0])
    extent_x = max_x - min_x
    
    max_y = np.max(point_cloud_np[:, 1])
    min_y = np.min(point_cloud_np[:, 1])
    extent_y = max_y - min_y
    crown_diameter = (extent_x + extent_y) / 2.0
    
    # 3. Canopy Density (ρ)
    # Proxy: Number of points divided by the volume of the bounding box
    num_points = point_cloud_np.shape[0]
    # Handle edge case where a dimension is zero
    volume = (extent_x + 1e-6) * (extent_y + 1e-6) * (tree_height + 1e-6)
    canopy_density = num_points / volume
    
    # Combine into F_struct vector
    # Note: These should be normalized across the *entire dataset*
    # Here, we return raw values; normalization should happen in the Dataset class
    
    traits = np.array([tree_height, crown_diameter, canopy_density], dtype=np.float32)
    
    # Handle potential NaNs or Infs
    traits[np.isnan(traits)] = 0.0
    traits[np.isinf(traits)] = 0.0
    
    return traits
