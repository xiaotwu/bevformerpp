"""
Spatial Fusion Module

This module implements spatial fusion between LiDAR and camera BEV features
using cross-attention mechanism as specified in the design document.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between LiDAR and camera BEV features.
    
    LiDAR features attend to camera features to incorporate semantic information
    while preserving geometric structure through residual connections.
    """
    
    def __init__(self, lidar_channels: int = 64, camera_channels: int = 256, 
                 fused_channels: int = 256, num_heads: int = 8):
        """
        Args:
            lidar_channels: Number of channels in LiDAR BEV features (C1)
            camera_channels: Number of channels in camera BEV features (C2)
            fused_channels: Number of channels in fused output (C3)
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels
        self.num_heads = num_heads
        
        # Project LiDAR features to fused dimension for query
        self.lidar_proj = nn.Conv2d(lidar_channels, fused_channels, kernel_size=1, bias=False)
        
        # Project camera features to fused dimension for key and value
        self.camera_key_proj = nn.Conv2d(camera_channels, fused_channels, kernel_size=1, bias=False)
        self.camera_value_proj = nn.Conv2d(camera_channels, fused_channels, kernel_size=1, bias=False)
        
        # Multi-head attention
        assert fused_channels % num_heads == 0, "fused_channels must be divisible by num_heads"
        self.head_dim = fused_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Output projection
        self.out_proj = nn.Conv2d(fused_channels, fused_channels, kernel_size=1, bias=False)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fused_channels)
        self.norm2 = nn.LayerNorm(fused_channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fused_channels, fused_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fused_channels * 4, fused_channels),
            nn.Dropout(0.1)
        )
        
        # Residual projection if dimensions don't match
        if lidar_channels != fused_channels:
            self.residual_proj = nn.Conv2d(lidar_channels, fused_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, F_lidar: torch.Tensor, F_cam: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of cross-attention fusion.
        
        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features
        
        Returns:
            F_fused: (B, C3, H, W) - Fused BEV features
        """
        B, C1, H, W = F_lidar.shape
        _, C2, H_cam, W_cam = F_cam.shape
        
        # Verify spatial dimensions match
        assert H == H_cam and W == W_cam, \
            f"Spatial dimensions must match: LiDAR ({H}, {W}) vs Camera ({H_cam}, {W_cam})"
        
        # Project features
        Q = self.lidar_proj(F_lidar)  # (B, C3, H, W)
        K = self.camera_key_proj(F_cam)  # (B, C3, H, W)
        V = self.camera_value_proj(F_cam)  # (B, C3, H, W)
        
        # Reshape for multi-head attention
        # (B, C3, H, W) -> (B, num_heads, H*W, head_dim)
        Q = Q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Compute attention scores
        # (B, num_heads, H*W, head_dim) @ (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, H*W)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, H*W, H*W)
        
        # Apply attention to values
        # (B, num_heads, H*W, H*W) @ (B, num_heads, H*W, head_dim) -> (B, num_heads, H*W, head_dim)
        attn_out = torch.matmul(attn_weights, V)
        
        # Reshape back to spatial format
        # (B, num_heads, H*W, head_dim) -> (B, C3, H, W)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, self.fused_channels, H, W)
        
        # Output projection
        attn_out = self.out_proj(attn_out)  # (B, C3, H, W)
        
        # First residual connection with layer norm
        # Reshape for layer norm: (B, C3, H, W) -> (B, H, W, C3)
        residual = self.residual_proj(F_lidar)
        x = residual + attn_out
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C3)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # (B, C3, H, W)
        
        # Feed-forward network with second residual connection
        # Reshape for FFN: (B, C3, H, W) -> (B, H*W, C3)
        identity = x
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, self.fused_channels)
        x = self.ffn(x)
        x = x.reshape(B, H, W, self.fused_channels).permute(0, 3, 1, 2)  # (B, C3, H, W)
        
        # Second residual connection with layer norm
        x = identity + x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C3)
        x = self.norm2(x)
        F_fused = x.permute(0, 3, 1, 2)  # (B, C3, H, W)
        
        return F_fused


class SpatialFusionModule(nn.Module):
    """
    Complete spatial fusion module that combines LiDAR and camera BEV features.
    
    This module ensures spatial alignment and applies cross-attention fusion
    to produce unified BEV features that preserve both geometric and semantic information.
    """
    
    def __init__(self, lidar_channels: int = 64, camera_channels: int = 256,
                 fused_channels: int = 256, num_heads: int = 8):
        """
        Args:
            lidar_channels: Number of channels in LiDAR BEV features (C1)
            camera_channels: Number of channels in camera BEV features (C2)
            fused_channels: Number of channels in fused output (C3)
            num_heads: Number of attention heads for cross-attention
        """
        super().__init__()
        
        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels
        
        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(
            lidar_channels=lidar_channels,
            camera_channels=camera_channels,
            fused_channels=fused_channels,
            num_heads=num_heads
        )
    
    def verify_alignment(self, F_lidar: torch.Tensor, F_cam: torch.Tensor) -> bool:
        """
        Verify that LiDAR and camera BEV features have matching spatial dimensions.
        
        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features
        
        Returns:
            bool: True if dimensions match, False otherwise
        """
        B_lidar, C_lidar, H_lidar, W_lidar = F_lidar.shape
        B_cam, C_cam, H_cam, W_cam = F_cam.shape
        
        # Check batch size
        if B_lidar != B_cam:
            return False
        
        # Check spatial dimensions
        if H_lidar != H_cam or W_lidar != W_cam:
            return False
        
        # Check channel dimensions match expected
        if C_lidar != self.lidar_channels or C_cam != self.camera_channels:
            return False
        
        return True
    
    def forward(self, F_lidar: torch.Tensor, F_cam: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of spatial fusion module.
        
        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features
        
        Returns:
            F_fused: (B, C3, H, W) - Fused BEV features
        
        Raises:
            AssertionError: If spatial dimensions don't match
        """
        # Verify alignment
        assert self.verify_alignment(F_lidar, F_cam), \
            f"Feature dimensions don't match: LiDAR {F_lidar.shape} vs Camera {F_cam.shape}"
        
        # Apply fusion
        F_fused = self.fusion(F_lidar, F_cam)
        
        return F_fused
