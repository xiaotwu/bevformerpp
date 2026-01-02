"""
Motion-Compensated Convolutional RNN (MC-ConvRNN) for temporal aggregation.
Implements ego-motion warping, motion field estimation, visibility gating, and ConvGRU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .utils import align_bev_features, compute_visibility_mask


class MotionFieldEstimator(nn.Module):
    """
    Lightweight CNN for estimating residual motion field.
    Estimates fine-grained motion beyond ego-motion compensation.
    
    The motion field represents pixel-wise 2D flow that captures
    dynamic object motion and ego-motion estimation errors.
    """
    
    def __init__(self, in_channels: int = 256, hidden_channels: int = 128):
        """
        Initialize motion field estimator.
        
        Args:
            in_channels: Number of input feature channels
            hidden_channels: Number of hidden channels in the network
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # Lightweight CNN: 3 convolutional layers
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
        
        # Output layer: predicts 2D flow field (dx, dy)
        self.conv3 = nn.Conv2d(hidden_channels // 2, 2, kernel_size=1)
        
        # Initialize output layer with small weights for stability
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
    
    def forward(self, current_features: torch.Tensor, 
                warped_features: torch.Tensor) -> torch.Tensor:
        """
        Estimate residual motion field between current and warped features.
        
        Args:
            current_features: Current BEV features, shape (B, C, H, W)
            warped_features: Ego-motion warped features, shape (B, C, H, W)
        
        Returns:
            Motion field (flow), shape (B, 2, H, W)
            Values represent (dx, dy) displacement in normalized coordinates [-1, 1]
        """
        # Concatenate current and warped features
        concat_features = torch.cat([current_features, warped_features], dim=1)
        
        # Pass through CNN
        x = self.conv1(concat_features)
        x = self.conv2(x)
        flow = self.conv3(x)
        
        # Optionally scale flow to reasonable range
        # Tanh bounds output to [-1, 1], then scale
        flow = torch.tanh(flow) * 0.1  # Small flow values
        
        return flow


def warp_with_flow(features: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """
    Warp features using optical flow field.
    
    Args:
        features: Features to warp, shape (B, C, H, W)
        flow: Flow field, shape (B, 2, H, W) with values in normalized coordinates
    
    Returns:
        Warped features, shape (B, C, H, W)
    """
    B, C, H, W = features.shape
    device = features.device
    
    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([grid_x, grid_y], dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
    
    # Add flow to base grid
    # flow is (B, 2, H, W), need to permute to (B, H, W, 2)
    flow_permuted = flow.permute(0, 2, 3, 1)  # (B, H, W, 2)
    sampling_grid = base_grid + flow_permuted
    
    # Warp features using grid sampling
    warped = F.grid_sample(
        features,
        sampling_grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    return warped


class VisibilityGating(nn.Module):
    """
    Apply visibility gating to warped features.
    Masks out regions that are out of bounds or occluded.
    """
    
    def __init__(self):
        """Initialize visibility gating module."""
        super().__init__()
    
    def forward(self, features: torch.Tensor, 
                visibility_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply visibility mask to features.
        
        Args:
            features: Features to gate, shape (B, C, H, W)
            visibility_mask: Visibility mask, shape (B, 1, H, W) with values in [0, 1]
        
        Returns:
            Gated features, shape (B, C, H, W)
        """
        return features * visibility_mask


class ConvGRUCell(nn.Module):
    """
    Convolutional GRU cell for spatial-temporal feature fusion.
    Implements update gate, reset gate, and hidden state update.
    """
    
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        """
        Initialize ConvGRU cell.
        
        Args:
            input_channels: Number of input feature channels
            hidden_channels: Number of hidden state channels
            kernel_size: Kernel size for convolutions
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Update gate
        self.conv_z = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Reset gate
        self.conv_r = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        
        # Candidate hidden state
        self.conv_h = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ConvGRU cell.
        
        Args:
            x: Input features, shape (B, C_in, H, W)
            h_prev: Previous hidden state, shape (B, C_h, H, W)
        
        Returns:
            New hidden state, shape (B, C_h, H, W)
        """
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Update gate
        z = torch.sigmoid(self.conv_z(combined))
        
        # Reset gate
        r = torch.sigmoid(self.conv_r(combined))
        
        # Candidate hidden state
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_reset))
        
        # New hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        
        return h_new
    
    def init_hidden(self, batch_size: int, height: int, width: int, 
                   device: torch.device) -> torch.Tensor:
        """
        Initialize hidden state with zeros.
        
        Args:
            batch_size: Batch size
            height: Spatial height
            width: Spatial width
            device: Device to create tensor on
        
        Returns:
            Initial hidden state, shape (B, C_h, H, W)
        """
        return torch.zeros(
            batch_size, 
            self.hidden_channels, 
            height, 
            width,
            device=device
        )


class MCConvRNN(nn.Module):
    """
    Motion-Compensated Convolutional RNN for temporal aggregation.

    Implements the complete 5-step process:
    1. Ego-motion warping
    2. Dynamic residual motion field estimation
    3. Visibility gating
    4. ConvGRU fusion
    5. Output projection

    Implements Requirements 5.1-5.5 from the design document.

    BPTT (Backpropagation Through Time) Behavior:
    ---------------------------------------------
    This module uses "truncated BPTT" by design:
    - The hidden state should be detached between sequences/scenes
    - For per-frame training, gradients flow through the current frame only
    - The caller is responsible for detaching hidden states at scene boundaries

    For full temporal gradient flow across multiple frames:
    - Pass the hidden state without detaching between forward calls
    - Be aware this increases memory usage linearly with sequence length
    - Consider gradient checkpointing for long sequences

    Example usage with truncated BPTT:
        hidden = None
        for sample in scene:
            output, hidden = mc_convrnn(features, prev_features, hidden, ego_motion)
            # hidden is reused across frames within a scene
        hidden = None  # Reset at scene boundary
    """
    
    def __init__(
        self,
        input_channels: int = 256,
        hidden_channels: int = 128,
        motion_hidden_channels: int = 128,
        kernel_size: int = 3,
        bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
        # Ablation study flags
        disable_warping: bool = False,
        disable_motion_field: bool = False,
        disable_visibility: bool = False
    ):
        """
        Initialize MC-ConvRNN module.

        Args:
            input_channels: Number of input feature channels
            hidden_channels: Number of hidden state channels for ConvGRU
            motion_hidden_channels: Hidden channels for motion field estimator
            kernel_size: Kernel size for ConvGRU convolutions
            bev_range: BEV range (x_min, x_max, y_min, y_max) in meters for warping
            disable_warping: Ablation - disable ego-motion warping
            disable_motion_field: Ablation - disable dynamic motion field estimation
            disable_visibility: Ablation - disable visibility gating
        """
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bev_range = bev_range
        
        # Ablation flags
        self.disable_warping = disable_warping
        self.disable_motion_field = disable_motion_field
        self.disable_visibility = disable_visibility
        
        # Motion field estimator
        self.motion_estimator = MotionFieldEstimator(
            in_channels=input_channels,
            hidden_channels=motion_hidden_channels
        )
        
        # Visibility gating
        self.visibility_gating = VisibilityGating()
        
        # ConvGRU cell
        self.convgru_cell = ConvGRUCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(input_channels + hidden_channels, input_channels, kernel_size=1),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        current_features: torch.Tensor,
        prev_features: Optional[torch.Tensor] = None,
        prev_hidden: Optional[torch.Tensor] = None,
        ego_motion: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of MC-ConvRNN.
        
        Args:
            current_features: Current BEV features, shape (B, C, H, W)
            prev_features: Previous BEV features, shape (B, C, H, W)
            prev_hidden: Previous hidden state, shape (B, C_h, H, W)
            ego_motion: Ego-motion transform, shape (B, 4, 4)
        
        Returns:
            Tuple of (output_features, new_hidden_state)
            - output_features: Motion-compensated features, shape (B, C, H, W)
            - new_hidden_state: Updated hidden state, shape (B, C_h, H, W)
        """
        B, C, H, W = current_features.shape
        device = current_features.device
        
        # Initialize hidden state if not provided
        if prev_hidden is None:
            prev_hidden = self.convgru_cell.init_hidden(B, H, W, device)
        
        # If no previous features, just use current features
        if prev_features is None:
            # First frame: no temporal aggregation
            new_hidden = self.convgru_cell(current_features, prev_hidden)
            output = self.output_proj(torch.cat([current_features, new_hidden], dim=1))
            return output, new_hidden
        
        # Step 1: Ego-motion warping (can be disabled for ablation)
        if ego_motion is not None and not self.disable_warping:
            warped_features = align_bev_features(prev_features, ego_motion, bev_range=self.bev_range)
            warped_hidden = align_bev_features(prev_hidden, ego_motion, bev_range=self.bev_range)
        else:
            warped_features = prev_features
            warped_hidden = prev_hidden
        
        # Step 2: Dynamic residual motion field (can be disabled for ablation)
        if not self.disable_motion_field:
            motion_field = self.motion_estimator(current_features, warped_features)
            aligned_features = warp_with_flow(warped_features, motion_field)
        else:
            aligned_features = warped_features
        
        # Step 3: Visibility gating (can be disabled for ablation)
        if not self.disable_visibility and ego_motion is not None:
            visibility_mask = compute_visibility_mask(ego_motion, H, W, bev_range=self.bev_range)
            gated_features = self.visibility_gating(aligned_features, visibility_mask)
        else:
            gated_features = aligned_features
        
        # Step 4: ConvGRU fusion
        # Fuse gated_features (motion-compensated history) with current_features
        # This creates an enhanced input that combines current observations with aligned history
        fused_input = current_features + gated_features  # Element-wise fusion
        
        # Use fused input with warped hidden state
        new_hidden = self.convgru_cell(fused_input, warped_hidden)
        
        # Step 5: Output projection
        output = self.output_proj(torch.cat([current_features, new_hidden], dim=1))
        
        return output, new_hidden
    
    def reset_hidden(self):
        """Reset hidden state (for scene boundaries)."""
        # Hidden state is managed externally, so this is a placeholder
        pass
