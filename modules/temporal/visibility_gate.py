"""
Visibility Gating Module for MC-ConvRNN.

Implements visibility-aware gating to mask out unreliable regions
in warped BEV features. This is critical for:
1. Masking out-of-bounds regions after ego-motion warping
2. Handling occlusion and newly revealed areas
3. Feature consistency-based confidence weighting

This is the third step in the MC-ConvRNN pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def compute_bounds_mask(
    ego_transform: torch.Tensor,
    H: int,
    W: int,
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)
) -> torch.Tensor:
    """
    Compute visibility mask based on grid sampling bounds.
    
    After ego-motion warping, some grid positions may sample from
    outside the valid [-1, 1] range. This function identifies those regions.
    
    Args:
        ego_transform: SE(3) transform (B, 4, 4)
        H: Grid height
        W: Grid width
        bev_range: (x_min, x_max, y_min, y_max) in meters
        
    Returns:
        Visibility mask (B, 1, H, W) with values in [0, 1]
    """
    B = ego_transform.shape[0]
    device = ego_transform.device
    dtype = ego_transform.dtype
    
    # Invert transformation for sampling
    try:
        ego_transform_inv = torch.inverse(ego_transform)
    except RuntimeError:
        # If inverse fails, return all-ones mask
        return torch.ones(B, 1, H, W, device=device, dtype=dtype)
    
    # Extract SE(2) components
    rotation_2d = ego_transform_inv[:, :2, :2]  # (B, 2, 2)
    translation_2d = ego_transform_inv[:, :2, 3]  # (B, 2)
    
    # Normalize translation
    x_min, x_max, y_min, y_max = bev_range
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    translation_normalized = torch.zeros(B, 2, device=device, dtype=dtype)
    translation_normalized[:, 0] = translation_2d[:, 0] * 2.0 / x_range
    translation_normalized[:, 1] = translation_2d[:, 1] * 2.0 / y_range
    
    # Build affine matrix
    affine_matrix = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    affine_matrix[:, :2, :2] = rotation_2d
    affine_matrix[:, :, 2] = translation_normalized
    
    # Generate sampling grid
    grid = F.affine_grid(affine_matrix, [B, 1, H, W], align_corners=False)
    
    # Check if grid points are within valid range [-1, 1]
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid_mask = (valid_x & valid_y).float()
    
    # Reshape to (B, 1, H, W)
    valid_mask = valid_mask.unsqueeze(1)
    
    return valid_mask


def compute_feature_consistency_mask(
    current_features: torch.Tensor,
    warped_features: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Compute visibility mask based on feature consistency.
    
    Regions where current and warped features differ significantly
    are likely unreliable (dynamic objects, occlusions, etc.).
    
    Args:
        current_features: Current BEV features (B, C, H, W)
        warped_features: Ego-warped features (B, C, H, W)
        threshold: Consistency threshold (lower = stricter)
        
    Returns:
        Consistency mask (B, 1, H, W) with values in [0, 1]
    """
    # Compute cosine similarity per position
    current_norm = F.normalize(current_features, dim=1, p=2)
    warped_norm = F.normalize(warped_features, dim=1, p=2)
    
    # Similarity: (B, H, W)
    similarity = (current_norm * warped_norm).sum(dim=1, keepdim=True)
    
    # Convert to soft mask
    # similarity is in [-1, 1], map to [0, 1] with threshold
    mask = torch.sigmoid((similarity - threshold) * 5.0)  # Soft threshold
    
    return mask


class VisibilityGate(nn.Module):
    """
    Visibility Gating Module.
    
    Computes a visibility mask that gates warped features to suppress
    unreliable regions. Supports multiple gating strategies:
    
    1. bounds: Mask out-of-bounds regions from ego-motion warping
    2. consistency: Mask regions with low feature consistency
    3. learned: Learn a gating function from features
    4. combined: Combine bounds and consistency masks
    
    The gate explicitly modulates the RNN update:
        h_new = visibility * f(warped) + (1 - visibility) * h_prev
    """
    
    def __init__(
        self,
        mode: str = "combined",
        in_channels: int = 256,
        consistency_threshold: float = 0.5,
        bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)
    ):
        """
        Args:
            mode: Gating mode - "bounds", "consistency", "learned", or "combined"
            in_channels: Feature channels (for learned mode)
            consistency_threshold: Threshold for consistency mask
            bev_range: BEV range for bounds computation
        """
        super().__init__()
        
        self.mode = mode
        self.in_channels = in_channels
        self.consistency_threshold = consistency_threshold
        self.bev_range = bev_range
        
        # Learned gating network (if mode is "learned")
        if mode == "learned":
            self.gate_net = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        current_features: torch.Tensor,
        warped_features: torch.Tensor,
        ego_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute visibility gate.
        
        Args:
            current_features: Current BEV features (B, C, H, W)
            warped_features: Ego-warped features (B, C, H, W)
            ego_transform: SE(3) transform (B, 4, 4), required for "bounds" mode
            
        Returns:
            Visibility gate (B, 1, H, W) with values in [0, 1]
        """
        B, C, H, W = current_features.shape
        device = current_features.device
        dtype = current_features.dtype
        
        if self.mode == "bounds":
            if ego_transform is None:
                return torch.ones(B, 1, H, W, device=device, dtype=dtype)
            return compute_bounds_mask(ego_transform, H, W, self.bev_range)
        
        elif self.mode == "consistency":
            return compute_feature_consistency_mask(
                current_features, warped_features, self.consistency_threshold
            )
        
        elif self.mode == "learned":
            concat = torch.cat([current_features, warped_features], dim=1)
            return self.gate_net(concat)
        
        elif self.mode == "combined":
            # Combine bounds and consistency masks
            bounds_mask = torch.ones(B, 1, H, W, device=device, dtype=dtype)
            if ego_transform is not None:
                bounds_mask = compute_bounds_mask(ego_transform, H, W, self.bev_range)
            
            consistency_mask = compute_feature_consistency_mask(
                current_features, warped_features, self.consistency_threshold
            )
            
            # Element-wise multiplication
            return bounds_mask * consistency_mask
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def apply_gate(
        self,
        features: torch.Tensor,
        gate: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply visibility gate to features.
        
        Args:
            features: Features to gate (B, C, H, W)
            gate: Visibility gate (B, 1, H, W)
            
        Returns:
            Gated features (B, C, H, W)
        """
        return features * gate
    
    def extra_repr(self) -> str:
        return f"mode={self.mode}, consistency_threshold={self.consistency_threshold}"


class VisibilityGatedUpdate(nn.Module):
    """
    Visibility-gated RNN update module.
    
    Implements the proposal-specified gated update:
        h_new = v * f(warped, current) + (1 - v) * h_prev
        
    where v is the visibility gate and f is the update function.
    """
    
    def __init__(
        self,
        hidden_channels: int,
        gate_mode: str = "combined",
        bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)
    ):
        """
        Args:
            hidden_channels: Hidden state channels
            gate_mode: Visibility gate mode
            bev_range: BEV range for bounds computation
        """
        super().__init__()
        
        self.visibility_gate = VisibilityGate(
            mode=gate_mode,
            in_channels=hidden_channels,
            bev_range=bev_range
        )
    
    def forward(
        self,
        new_state: torch.Tensor,
        prev_state: torch.Tensor,
        current_features: torch.Tensor,
        warped_features: torch.Tensor,
        ego_transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply visibility-gated update.
        
        Args:
            new_state: Proposed new hidden state (B, C, H, W)
            prev_state: Previous hidden state (B, C, H, W)
            current_features: Current BEV features (B, C, H, W)
            warped_features: Warped features (B, C, H, W)
            ego_transform: Ego-motion transform (B, 4, 4)
            
        Returns:
            Updated hidden state (B, C, H, W)
        """
        # Compute visibility gate
        v = self.visibility_gate(current_features, warped_features, ego_transform)
        
        # Gated update
        h_new = v * new_state + (1 - v) * prev_state
        
        return h_new

