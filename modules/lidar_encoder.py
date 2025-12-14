"""
LiDAR BEV Encoder using PointPillars architecture.
Converts point clouds to BEV feature representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

from modules.data_structures import BEVGridConfig


class Pillarization:
    """
    Converts point cloud to pillar representation.
    Handles point-to-pillar assignment and coordinate computation.
    """
    
    def __init__(self, config: BEVGridConfig, max_points_per_pillar: int = 100, max_pillars: int = 12000):
        """
        Args:
            config: BEV grid configuration
            max_points_per_pillar: Maximum number of points to keep per pillar
            max_pillars: Maximum number of pillars to process
        """
        self.config = config
        self.max_points_per_pillar = max_points_per_pillar
        self.max_pillars = max_pillars
        
        # Compute grid parameters
        self.H, self.W = config.grid_size
        self.x_min, self.x_max = config.x_range
        self.y_min, self.y_max = config.y_range
        self.z_min, self.z_max = config.z_range
        self.resolution = config.resolution
    
    def __call__(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert point cloud to pillar representation.
        
        Args:
            points: (N, 4) array with columns [x, y, z, intensity]
        
        Returns:
            pillars: (num_pillars, max_points_per_pillar, 4) array of pillar points
            pillar_coords: (num_pillars, 2) array of pillar coordinates [y_idx, x_idx]
            num_points_per_pillar: (num_pillars,) array of actual point counts
        """
        # Filter points within BEV bounds
        mask = (
            (points[:, 0] >= self.x_min) & (points[:, 0] < self.x_max) &
            (points[:, 1] >= self.y_min) & (points[:, 1] < self.y_max) &
            (points[:, 2] >= self.z_min) & (points[:, 2] < self.z_max)
        )
        points = points[mask]
        
        if len(points) == 0:
            # Return empty pillars
            pillars = np.zeros((0, self.max_points_per_pillar, 4), dtype=np.float32)
            pillar_coords = np.zeros((0, 2), dtype=np.int32)
            num_points_per_pillar = np.zeros((0,), dtype=np.int32)
            return pillars, pillar_coords, num_points_per_pillar
        
        # Compute pillar indices for each point
        x_idx = ((points[:, 0] - self.x_min) / self.resolution).astype(np.int32)
        y_idx = ((points[:, 1] - self.y_min) / self.resolution).astype(np.int32)
        
        # Clip to valid range (should not be necessary after filtering, but for safety)
        x_idx = np.clip(x_idx, 0, self.W - 1)
        y_idx = np.clip(y_idx, 0, self.H - 1)
        
        # Create pillar indices (flatten 2D grid to 1D)
        pillar_indices = y_idx * self.W + x_idx
        
        # Group points by pillar
        unique_pillars, inverse_indices = np.unique(pillar_indices, return_inverse=True)
        num_unique_pillars = len(unique_pillars)
        
        # Limit number of pillars
        if num_unique_pillars > self.max_pillars:
            # Keep only the first max_pillars
            unique_pillars = unique_pillars[:self.max_pillars]
            num_unique_pillars = self.max_pillars
            # Filter points to only those in kept pillars
            mask = np.isin(pillar_indices, unique_pillars)
            points = points[mask]
            pillar_indices = pillar_indices[mask]
            _, inverse_indices = np.unique(pillar_indices, return_inverse=True)
        
        # Initialize output arrays
        pillars = np.zeros((num_unique_pillars, self.max_points_per_pillar, 4), dtype=np.float32)
        num_points_per_pillar = np.zeros((num_unique_pillars,), dtype=np.int32)
        
        # Fill pillars with points
        for i in range(num_unique_pillars):
            # Get points belonging to this pillar
            pillar_mask = inverse_indices == i
            pillar_points = points[pillar_mask]
            
            # Limit points per pillar
            num_points = min(len(pillar_points), self.max_points_per_pillar)
            pillars[i, :num_points] = pillar_points[:num_points]
            num_points_per_pillar[i] = num_points
        
        # Convert pillar indices back to 2D coordinates
        pillar_y = unique_pillars // self.W
        pillar_x = unique_pillars % self.W
        pillar_coords = np.stack([pillar_y, pillar_x], axis=1).astype(np.int32)
        
        return pillars, pillar_coords, num_points_per_pillar


class PillarFeatureNet(nn.Module):
    """
    PointNet-style encoder for pillar features.
    Encodes each pillar independently using a shared MLP.
    """
    
    def __init__(self, in_channels: int = 9, out_channels: int = 64):
        """
        Args:
            in_channels: Number of input features per point (default 9: x, y, z, r, xc, yc, zc, xp, yp)
            out_channels: Number of output features per pillar
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Shared MLP for point encoding
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, pillars: torch.Tensor, pillar_coords: torch.Tensor, 
                num_points_per_pillar: torch.Tensor, config: BEVGridConfig) -> torch.Tensor:
        """
        Encode pillar features.
        
        Args:
            pillars: (num_pillars, max_points_per_pillar, 4) tensor [x, y, z, intensity]
            pillar_coords: (num_pillars, 2) tensor [y_idx, x_idx]
            num_points_per_pillar: (num_pillars,) tensor of point counts
            config: BEV grid configuration
        
        Returns:
            pillar_features: (num_pillars, out_channels) tensor
        """
        num_pillars, max_points, _ = pillars.shape
        device = pillars.device
        
        if num_pillars == 0:
            return torch.zeros((0, self.out_channels), device=device, dtype=pillars.dtype)
        
        # Compute pillar centers in metric coordinates
        pillar_x_center = (pillar_coords[:, 1].float() + 0.5) * config.resolution + config.x_min
        pillar_y_center = (pillar_coords[:, 0].float() + 0.5) * config.resolution + config.y_min
        
        # Compute pillar-wise mean z
        # Create mask for valid points
        point_mask = torch.arange(max_points, device=device)[None, :] < num_points_per_pillar[:, None]
        
        # Compute mean z for each pillar
        z_values = pillars[:, :, 2]  # (num_pillars, max_points)
        z_values_masked = z_values * point_mask.float()
        pillar_z_mean = z_values_masked.sum(dim=1) / num_points_per_pillar.float().clamp(min=1)
        
        # Augment point features with relative coordinates
        # For each point: [x, y, z, r, xc, yc, zc, xp, yp]
        # xc, yc, zc: offset from pillar center
        # xp, yp: offset from pillar center in grid coordinates
        
        x_offset = pillars[:, :, 0] - pillar_x_center[:, None]  # (num_pillars, max_points)
        y_offset = pillars[:, :, 1] - pillar_y_center[:, None]
        z_offset = pillars[:, :, 2] - pillar_z_mean[:, None]
        
        # Stack augmented features
        augmented_features = torch.stack([
            pillars[:, :, 0],  # x
            pillars[:, :, 1],  # y
            pillars[:, :, 2],  # z
            pillars[:, :, 3],  # intensity
            x_offset,          # xc
            y_offset,          # yc
            z_offset,          # zc
            x_offset,          # xp (same as xc for simplicity)
            y_offset           # yp (same as yc for simplicity)
        ], dim=-1)  # (num_pillars, max_points, 9)
        
        # Reshape for MLP: (num_pillars * max_points, 9)
        features_flat = augmented_features.reshape(-1, self.in_channels)
        
        # Apply MLP
        encoded_flat = self.mlp(features_flat)  # (num_pillars * max_points, out_channels)
        
        # Reshape back: (num_pillars, max_points, out_channels)
        encoded = encoded_flat.reshape(num_pillars, max_points, self.out_channels)
        
        # Max pooling over points in each pillar
        # Mask out invalid points before max pooling
        encoded_masked = encoded * point_mask[:, :, None].float()
        encoded_masked = encoded_masked + (1 - point_mask[:, :, None].float()) * (-1e9)  # Large negative for invalid
        pillar_features = encoded_masked.max(dim=1)[0]  # (num_pillars, out_channels)
        
        return pillar_features


class PointPillarsScatter(nn.Module):
    """
    Scatters pillar features to dense BEV grid.
    """
    
    def __init__(self, config: BEVGridConfig, in_channels: int = 64):
        """
        Args:
            config: BEV grid configuration
            in_channels: Number of input channels per pillar
        """
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.H, self.W = config.grid_size
    
    def forward(self, pillar_features: torch.Tensor, pillar_coords: torch.Tensor, 
                batch_size: int = 1) -> torch.Tensor:
        """
        Scatter pillar features to BEV grid.
        
        Args:
            pillar_features: (num_pillars, in_channels) tensor
            pillar_coords: (num_pillars, 2) tensor [y_idx, x_idx]
            batch_size: Batch size (for now, assumes single batch)
        
        Returns:
            bev_features: (batch_size, in_channels, H, W) tensor
        """
        device = pillar_features.device
        
        # Initialize BEV grid with zeros
        bev_features = torch.zeros((batch_size, self.in_channels, self.H, self.W), 
                                   device=device, dtype=pillar_features.dtype)
        
        if pillar_features.shape[0] == 0:
            return bev_features
        
        # Scatter features to grid
        # For batch_idx=0 (single batch for now)
        batch_idx = 0
        y_coords = pillar_coords[:, 0].long()
        x_coords = pillar_coords[:, 1].long()
        
        # Place features at corresponding locations
        bev_features[batch_idx, :, y_coords, x_coords] = pillar_features.t()
        
        return bev_features


class BackboneCNN(nn.Module):
    """
    2D CNN backbone for BEV feature refinement.
    Uses residual blocks for progressive feature extraction.
    """
    
    def __init__(self, in_channels: int = 64, out_channels: int = 64):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        
        # Simple residual backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.res_block1 = self._make_residual_block(64, 64)
        self.res_block2 = self._make_residual_block(64, 64)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_residual_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels, H, W) tensor
        
        Returns:
            out: (B, out_channels, H, W) tensor
        """
        x = self.conv1(x)
        
        # Residual block 1
        identity = x
        x = self.res_block1(x)
        x = F.relu(x + identity, inplace=True)
        
        # Residual block 2
        identity = x
        x = self.res_block2(x)
        x = F.relu(x + identity, inplace=True)
        
        x = self.conv_out(x)
        
        return x


class LiDARBEVEncoder(nn.Module):
    """
    Complete LiDAR BEV encoder using PointPillars architecture.
    Converts point clouds to BEV feature representations.
    """
    
    def __init__(self, config: BEVGridConfig, out_channels: int = 64,
                 max_points_per_pillar: int = 100, max_pillars: int = 12000):
        """
        Args:
            config: BEV grid configuration
            out_channels: Number of output feature channels (C1)
            max_points_per_pillar: Maximum points per pillar
            max_pillars: Maximum number of pillars
        """
        super().__init__()
        self.config = config
        self.out_channels = out_channels
        
        # Pillarization (not a nn.Module, just a callable)
        self.pillarization = Pillarization(config, max_points_per_pillar, max_pillars)
        
        # Feature extraction
        self.pillar_feature_net = PillarFeatureNet(in_channels=9, out_channels=64)
        
        # Scatter to BEV
        self.scatter = PointPillarsScatter(config, in_channels=64)
        
        # 2D CNN backbone
        self.backbone = BackboneCNN(in_channels=64, out_channels=out_channels)
    
    def forward(self, points_batch: List[np.ndarray]) -> torch.Tensor:
        """
        Forward pass for batch of point clouds.
        
        Args:
            points_batch: List of (N_i, 4) numpy arrays [x, y, z, intensity]
        
        Returns:
            bev_features: (B, C1, H, W) tensor
        """
        batch_size = len(points_batch)
        device = next(self.parameters()).device
        
        # Process each point cloud in batch
        bev_features_list = []
        
        for points in points_batch:
            # Pillarization
            pillars, pillar_coords, num_points = self.pillarization(points)
            
            # Convert to tensors
            pillars_t = torch.from_numpy(pillars).to(device)
            pillar_coords_t = torch.from_numpy(pillar_coords).to(device)
            num_points_t = torch.from_numpy(num_points).to(device)
            
            # Feature extraction
            pillar_features = self.pillar_feature_net(pillars_t, pillar_coords_t, 
                                                     num_points_t, self.config)
            
            # Scatter to BEV
            bev = self.scatter(pillar_features, pillar_coords_t, batch_size=1)
            
            bev_features_list.append(bev)
        
        # Stack batch
        bev_features = torch.cat(bev_features_list, dim=0)  # (B, 64, H, W)
        
        # Apply backbone
        bev_features = self.backbone(bev_features)  # (B, C1, H, W)
        
        return bev_features
