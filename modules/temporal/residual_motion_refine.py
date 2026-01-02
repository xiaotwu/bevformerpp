"""
Residual Motion Refinement Module for MC-ConvRNN.

Implements dynamic residual motion field estimation to refine ego-motion warping.
This captures:
1. Ego-motion estimation errors
2. Dynamic object motion
3. Deformable alignment

This is the second step in the MC-ConvRNN pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMotionRefine(nn.Module):
    """
    Lightweight CNN for estimating residual motion field.
    
    Given current BEV features and ego-motion warped features,
    predicts a small offset field (Δu, Δv) to refine the alignment.
    
    Architecture:
    - 3 conv layers with BN and ReLU
    - Output: 2-channel flow field in normalized coordinates
    - Initialized to near-zero for stable training start
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        max_offset: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden channels
            max_offset: Maximum flow magnitude in normalized coords (tanh scaling)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.max_offset = max_offset
        
        # Lightweight CNN: concatenate current and warped features
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # Output: 2D flow field (Δu, Δv)
        self.conv3 = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)
        
        # Initialize to near-zero for stable training
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
    
    def forward(
        self,
        current_features: torch.Tensor,
        warped_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate residual motion field.
        
        Args:
            current_features: Current BEV features (B, C, H, W)
            warped_features: Ego-motion warped features (B, C, H, W)
            
        Returns:
            Flow field (B, 2, H, W) with values in [-max_offset, max_offset]
        """
        # Concatenate features
        concat = torch.cat([current_features, warped_features], dim=1)
        
        # Pass through CNN
        x = self.conv1(concat)
        x = self.conv2(x)
        flow = self.conv3(x)
        
        # Bound output with tanh and scale
        flow = torch.tanh(flow) * self.max_offset
        
        return flow
    
    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, hidden_channels={self.hidden_channels}, max_offset={self.max_offset}"


def apply_residual_warp(
    features: torch.Tensor,
    flow: torch.Tensor
) -> torch.Tensor:
    """
    Apply residual flow field to warp features.
    
    Args:
        features: Features to warp (B, C, H, W)
        flow: Flow field (B, 2, H, W) in normalized coordinates
        
    Returns:
        Warped features (B, C, H, W)
    """
    B, C, H, W = features.shape
    device = features.device
    
    # Create base grid in normalized coordinates [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    
    # Add flow to base grid
    flow_permuted = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
    sampling_grid = base_grid + flow_permuted
    
    # Warp features
    warped = F.grid_sample(
        features,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    return warped


class ResidualMotionModule(nn.Module):
    """
    Complete residual motion refinement module.
    
    Combines flow estimation and warping into a single module.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        hidden_channels: int = 128,
        max_offset: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden channels for flow estimation
            max_offset: Maximum flow magnitude
        """
        super().__init__()
        
        self.flow_estimator = ResidualMotionRefine(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            max_offset=max_offset
        )
    
    def forward(
        self,
        current_features: torch.Tensor,
        warped_features: torch.Tensor,
        return_flow: bool = False
    ):
        """
        Estimate and apply residual motion.
        
        Args:
            current_features: Current BEV features (B, C, H, W)
            warped_features: Ego-warped features (B, C, H, W)
            return_flow: If True, also return the flow field
            
        Returns:
            refined_features: Refined warped features (B, C, H, W)
            flow (optional): Flow field (B, 2, H, W) if return_flow=True
        """
        # Estimate flow
        flow = self.flow_estimator(current_features, warped_features)
        
        # Apply warp
        refined = apply_residual_warp(warped_features, flow)
        
        if return_flow:
            return refined, flow
        return refined

