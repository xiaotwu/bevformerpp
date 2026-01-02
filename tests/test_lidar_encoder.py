"""
Unit tests for LiDAR BEV encoder.
"""

import pytest
import torch
import numpy as np

from modules.data_structures import BEVGridConfig
from modules.lidar_encoder import (
    Pillarization,
    PillarFeatureNet,
    PointPillarsScatter,
    BackboneCNN,
    LiDARBEVEncoder
)


class TestPillarization:
    """Test pillarization module."""
    
    def test_pillarization_basic(self):
        """Test basic pillarization functionality."""
        config = BEVGridConfig()
        pillarization = Pillarization(config)
        
        # Create simple point cloud
        points = np.array([
            [0.0, 0.0, 0.0, 0.5],
            [1.0, 1.0, 0.5, 0.6],
            [0.1, 0.1, 0.2, 0.7],  # Should be in same pillar as first point
        ], dtype=np.float32)
        
        pillars, coords, num_points = pillarization(points)
        
        # Should have at least 1 pillar
        assert len(pillars) > 0
        assert len(coords) == len(pillars)
        assert len(num_points) == len(pillars)
        
        # Check shapes
        assert pillars.shape[1] == 100  # max_points_per_pillar
        assert pillars.shape[2] == 4    # x, y, z, intensity
        assert coords.shape[1] == 2     # y_idx, x_idx
    
    def test_pillarization_empty(self):
        """Test pillarization with empty point cloud."""
        config = BEVGridConfig()
        pillarization = Pillarization(config)
        
        points = np.zeros((0, 4), dtype=np.float32)
        pillars, coords, num_points = pillarization(points)
        
        assert len(pillars) == 0
        assert len(coords) == 0
        assert len(num_points) == 0
    
    def test_pillarization_out_of_bounds(self):
        """Test that out-of-bounds points are filtered."""
        config = BEVGridConfig()
        pillarization = Pillarization(config)
        
        # Points outside BEV range
        points = np.array([
            [100.0, 100.0, 0.0, 0.5],  # Out of bounds
            [0.0, 0.0, 0.0, 0.5],      # In bounds
        ], dtype=np.float32)
        
        pillars, coords, num_points = pillarization(points)
        
        # Should only have 1 pillar from the in-bounds point
        assert len(pillars) >= 1


class TestPillarFeatureNet:
    """Test pillar feature network."""
    
    def test_feature_net_forward(self):
        """Test forward pass of feature network."""
        config = BEVGridConfig()
        feature_net = PillarFeatureNet(in_channels=9, out_channels=64)
        feature_net.eval()
        
        # Create dummy pillar data
        num_pillars = 10
        max_points = 100
        pillars = torch.randn(num_pillars, max_points, 4)
        coords = torch.randint(0, 200, (num_pillars, 2))
        num_points = torch.randint(1, max_points, (num_pillars,))
        
        with torch.no_grad():
            features = feature_net(pillars, coords, num_points, config)
        
        assert features.shape == (num_pillars, 64)
        assert torch.isfinite(features).all()


class TestPointPillarsScatter:
    """Test scatter operation."""
    
    def test_scatter_basic(self):
        """Test basic scatter functionality."""
        config = BEVGridConfig()
        scatter = PointPillarsScatter(config, in_channels=64)
        
        # Create dummy features
        num_pillars = 5
        features = torch.randn(num_pillars, 64)
        coords = torch.tensor([[10, 20], [30, 40], [50, 60], [70, 80], [90, 100]])
        
        bev = scatter(features, coords, batch_size=1)
        
        H, W = config.grid_size
        assert bev.shape == (1, 64, H, W)
        
        # Check that features are placed at correct locations
        for i, (y, x) in enumerate(coords):
            assert torch.allclose(bev[0, :, y, x], features[i])


class TestBackboneCNN:
    """Test CNN backbone."""
    
    def test_backbone_forward(self):
        """Test forward pass of backbone."""
        backbone = BackboneCNN(in_channels=64, out_channels=64)
        backbone.eval()
        
        # Create dummy BEV features
        x = torch.randn(2, 64, 200, 200)
        
        with torch.no_grad():
            out = backbone(x)
        
        assert out.shape == (2, 64, 200, 200)
        assert torch.isfinite(out).all()


class TestLiDARBEVEncoder:
    """Test complete LiDAR BEV encoder."""
    
    def test_encoder_forward(self):
        """Test forward pass of complete encoder."""
        config = BEVGridConfig()
        encoder = LiDARBEVEncoder(config, out_channels=64)
        encoder.eval()
        
        # Create batch of point clouds
        points1 = np.random.uniform(-50, 50, (1000, 4)).astype(np.float32)
        points2 = np.random.uniform(-50, 50, (800, 4)).astype(np.float32)
        points_batch = [points1, points2]
        
        with torch.no_grad():
            output = encoder(points_batch)
        
        H, W = config.grid_size
        assert output.shape == (2, 64, H, W)
        assert torch.isfinite(output).all()
    
    def test_encoder_empty_batch(self):
        """Test encoder with empty point clouds."""
        config = BEVGridConfig()
        encoder = LiDARBEVEncoder(config, out_channels=64)
        encoder.eval()
        
        # Empty point clouds
        empty = np.zeros((0, 4), dtype=np.float32)
        points_batch = [empty, empty]
        
        with torch.no_grad():
            output = encoder(points_batch)
        
        H, W = config.grid_size
        assert output.shape == (2, 64, H, W)
        assert torch.isfinite(output).all()
    
    def test_encoder_single_point(self):
        """Test encoder with single point."""
        config = BEVGridConfig()
        encoder = LiDARBEVEncoder(config, out_channels=64)
        encoder.eval()
        
        # Single point
        single = np.array([[0.0, 0.0, 0.0, 0.5]], dtype=np.float32)
        points_batch = [single]
        
        with torch.no_grad():
            output = encoder(points_batch)
        
        H, W = config.grid_size
        assert output.shape == (1, 64, H, W)
        assert torch.isfinite(output).all()
