"""
Ego-Motion Warping Module for MC-ConvRNN.

Implements SE(2) BEV warping using ego-motion transforms.
This is the first step in the MC-ConvRNN pipeline.

Coordinate Conventions:
- Input ego_transform: SE(3) matrix (B, 4, 4) representing T_{prev->curr}
- For warping, we need the inverse T_{curr->prev} to sample from previous frame
- BEV coordinates: x-axis forward, y-axis left, z-axis up (NuScenes convention)
- Grid sampling uses normalized coordinates [-1, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def create_bev_grid(
    H: int,
    W: int,
    bev_range: Tuple[float, float, float, float],
    device: torch.device,
    dtype: torch.dtype
) -> torch.Tensor:
    """
    Create a BEV coordinate grid in meters.
    
    Args:
        H: Grid height
        W: Grid width
        bev_range: (x_min, x_max, y_min, y_max) in meters
        device: Target device
        dtype: Target dtype
        
    Returns:
        Grid of shape (H, W, 2) with (x, y) coordinates in meters
    """
    x_min, x_max, y_min, y_max = bev_range
    
    # Create coordinate vectors
    x_coords = torch.linspace(x_min, x_max, W, device=device, dtype=dtype)
    y_coords = torch.linspace(y_min, y_max, H, device=device, dtype=dtype)
    
    # Create meshgrid (y corresponds to rows, x to columns)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Stack to (H, W, 2) with (x, y) order
    grid = torch.stack([xx, yy], dim=-1)
    
    return grid


def warp_bev_with_ego_motion(
    features: torch.Tensor,
    ego_transform: torch.Tensor,
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)
) -> torch.Tensor:
    """
    Warp BEV features using ego-motion transformation.
    
    This function aligns features from the previous frame to the current frame
    by applying the inverse ego-motion transform.
    
    Args:
        features: BEV features to warp (B, C, H, W)
        ego_transform: SE(3) transform from prev to curr frame (B, 4, 4)
        bev_range: (x_min, x_max, y_min, y_max) in meters
        
    Returns:
        Warped features aligned to current frame (B, C, H, W)
    """
    B, C, H, W = features.shape
    device = features.device
    dtype = features.dtype
    
    # Invert the transformation: T_{curr->prev}
    try:
        ego_transform_inv = torch.inverse(ego_transform)
    except RuntimeError:
        # Singular matrix - return unchanged
        return features
    
    # Extract SE(2) components from SE(3)
    # Rotation: 2x2 block from top-left of rotation matrix
    # Translation: x, y components
    rotation_2d = ego_transform_inv[:, :2, :2]  # (B, 2, 2)
    translation_2d = ego_transform_inv[:, :2, 3]  # (B, 2)
    
    # Normalize translation for grid_sample coordinates
    x_min, x_max, y_min, y_max = bev_range
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Translation in normalized [-1, 1] coordinates
    translation_normalized = torch.zeros(B, 2, device=device, dtype=dtype)
    translation_normalized[:, 0] = translation_2d[:, 0] * 2.0 / x_range
    translation_normalized[:, 1] = translation_2d[:, 1] * 2.0 / y_range
    
    # Build affine matrix for grid_sample
    affine_matrix = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    affine_matrix[:, :2, :2] = rotation_2d
    affine_matrix[:, :, 2] = translation_normalized
    
    # Generate sampling grid
    grid = F.affine_grid(affine_matrix, [B, C, H, W], align_corners=False)
    
    # Warp features
    warped = F.grid_sample(
        features,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    return warped


class EgoMotionWarp(nn.Module):
    """
    Ego-Motion Warping Module.
    
    Wraps the warping function as an nn.Module for integration
    into the MC-ConvRNN pipeline.
    
    This module applies SE(2) (x, y, yaw) warping to align BEV features
    from a previous frame to the current frame using the ego-motion transform.
    """
    
    def __init__(self, bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)):
        """
        Args:
            bev_range: (x_min, x_max, y_min, y_max) in meters
        """
        super().__init__()
        self.bev_range = bev_range
    
    def forward(
        self,
        features: torch.Tensor,
        ego_transform: torch.Tensor
    ) -> torch.Tensor:
        """
        Warp features using ego-motion.
        
        Args:
            features: BEV features (B, C, H, W)
            ego_transform: SE(3) transform (B, 4, 4)
            
        Returns:
            Warped features (B, C, H, W)
        """
        return warp_bev_with_ego_motion(features, ego_transform, self.bev_range)
    
    def extra_repr(self) -> str:
        return f"bev_range={self.bev_range}"

