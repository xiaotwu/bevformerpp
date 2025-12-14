"""
Utility functions for BEV Fusion System.
Includes ego-motion warping, coordinate transformations, and other helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


def align_bev_features(features: torch.Tensor, ego_transform: torch.Tensor) -> torch.Tensor:
    """
    Warp BEV features from previous frame to current frame using ego-motion.
    Uses bilinear interpolation for smooth warping.
    
    Args:
        features: BEV features from previous frame, shape (B, C, H, W)
        ego_transform: SE(3) transformation matrix from previous to current frame,
                      shape (B, 4, 4)
    
    Returns:
        Warped BEV features aligned to current frame, shape (B, C, H, W)
    """
    B, C, H, W = features.shape
    device = features.device
    
    # Extract 2D transformation (x, y, yaw) from SE(3) matrix
    # ego_transform is T_{prev->curr}
    # For grid_sample, we need the inverse: T_{curr->prev}
    # This tells us where to sample from in the previous frame for each current frame pixel
    
    try:
        # Invert the transformation
        ego_transform_inv = torch.inverse(ego_transform)
    except RuntimeError:
        # If matrix is singular, return features unchanged
        return features
    
    # Extract 2D affine transformation (top-left 2x3 of the 4x4 matrix)
    # This handles translation and rotation in the BEV plane
    affine_matrix = ego_transform_inv[:, :2, :3]  # (B, 2, 3)
    
    # Generate sampling grid
    # affine_grid expects theta of shape (B, 2, 3)
    grid = F.affine_grid(affine_matrix, [B, C, H, W], align_corners=False)
    
    # Warp features using bilinear interpolation
    warped_features = F.grid_sample(
        features, 
        grid, 
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    return warped_features


def generate_grid_from_transform(ego_transform: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Generate sampling grid from ego-motion transformation.
    
    Args:
        ego_transform: SE(3) transformation matrix, shape (B, 4, 4)
        H: Height of BEV grid
        W: Width of BEV grid
    
    Returns:
        Sampling grid of shape (B, H, W, 2)
    """
    B = ego_transform.shape[0]
    device = ego_transform.device
    
    # Invert transformation for sampling
    ego_transform_inv = torch.inverse(ego_transform)
    
    # Extract 2D affine part
    affine_matrix = ego_transform_inv[:, :2, :3]  # (B, 2, 3)
    
    # Generate grid
    grid = F.affine_grid(affine_matrix, [B, 1, H, W], align_corners=False)
    
    return grid


def compute_visibility_mask(ego_transform: torch.Tensor, H: int, W: int, 
                            bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)) -> torch.Tensor:
    """
    Compute visibility mask for warped BEV features.
    Marks pixels that are out of bounds or occluded as invalid.
    
    Args:
        ego_transform: SE(3) transformation matrix, shape (B, 4, 4)
        H: Height of BEV grid
        W: Width of BEV grid
        bev_range: Tuple of (x_min, x_max, y_min, y_max) in meters
    
    Returns:
        Visibility mask of shape (B, 1, H, W) with values in [0, 1]
    """
    B = ego_transform.shape[0]
    device = ego_transform.device
    
    # Generate grid in normalized coordinates [-1, 1]
    grid = generate_grid_from_transform(ego_transform, H, W)  # (B, H, W, 2)
    
    # Check if grid points are within valid range [-1, 1]
    # Points outside this range are out of bounds after warping
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid_mask = valid_x & valid_y  # (B, H, W)
    
    # Convert to float and add channel dimension
    visibility_mask = valid_mask.float().unsqueeze(1)  # (B, 1, H, W)
    
    return visibility_mask


def warp_bev(prev_bev: torch.Tensor, ego_motion: torch.Tensor) -> torch.Tensor:
    """
    Legacy function name for backward compatibility.
    Wraps align_bev_features.
    
    Args:
        prev_bev: Previous BEV features, shape (B, C, H, W)
        ego_motion: Ego-motion transformation, shape (B, 4, 4)
    
    Returns:
        Warped BEV features, shape (B, C, H, W)
    """
    return align_bev_features(prev_bev, ego_motion)
