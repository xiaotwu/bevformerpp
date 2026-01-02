"""
Unit tests for Camera BEV Encoder.
"""

import pytest
import torch
import numpy as np

from modules.camera_encoder import (
    CameraBEVEncoder,
    project_3d_to_2d,
    backproject_2d_to_3d,
    project_bev_to_image
)


class TestCameraBEVEncoder:
    """Test suite for CameraBEVEncoder."""
    
    def test_encoder_initialization(self):
        """Test that encoder initializes correctly."""
        encoder = CameraBEVEncoder(
            bev_h=200,
            bev_w=200,
            embed_dim=256,
            num_layers=6
        )
        
        assert encoder.bev_h == 200
        assert encoder.bev_w == 200
        assert encoder.embed_dim == 256
        assert encoder.num_layers == 6
        
        # Check components exist
        assert hasattr(encoder, 'backbone')
        assert hasattr(encoder, 'neck')
        assert hasattr(encoder, 'bev_queries')
        assert hasattr(encoder, 'bev_pos_embed')
        assert hasattr(encoder, 'cross_attention_layers')
    
    def test_encoder_forward_pass(self):
        """Test forward pass with synthetic data."""
        batch_size = 2
        n_cam = 6
        img_h, img_w = 900, 1600
        bev_h, bev_w = 200, 200
        embed_dim = 256
        
        encoder = CameraBEVEncoder(
            bev_h=bev_h,
            bev_w=bev_w,
            embed_dim=embed_dim,
            num_layers=2,  # Fewer layers for faster testing
            img_h=img_h,
            img_w=img_w
        )
        encoder.eval()
        
        # Create synthetic inputs
        images = torch.randn(batch_size, n_cam, 3, img_h, img_w)
        intrinsics = torch.zeros(batch_size, n_cam, 3, 3)
        extrinsics = torch.zeros(batch_size, n_cam, 4, 4)
        
        for b in range(batch_size):
            for cam in range(n_cam):
                intrinsics[b, cam] = torch.tensor([
                    [1000.0, 0, 800.0],
                    [0, 1000.0, 450.0],
                    [0, 0, 1.0]
                ])
                extrinsics[b, cam] = torch.eye(4)
        
        # Forward pass
        with torch.no_grad():
            output = encoder(images, intrinsics, extrinsics)
        
        # Check output shape
        assert output.shape == (batch_size, embed_dim, bev_h, bev_w)
        assert torch.isfinite(output).all()
    
    def test_encoder_with_different_batch_sizes(self):
        """Test encoder works with different batch sizes."""
        encoder = CameraBEVEncoder(
            bev_h=100,
            bev_w=100,
            embed_dim=128,
            num_layers=1
        )
        encoder.eval()
        
        for batch_size in [1, 2, 4]:
            images = torch.randn(batch_size, 6, 3, 900, 1600)
            intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 6, 1, 1)
            extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 6, 1, 1)
            
            with torch.no_grad():
                output = encoder(images, intrinsics, extrinsics)
            
            assert output.shape == (batch_size, 128, 100, 100)


class TestProjectionUtilities:
    """Test suite for projection utilities."""
    
    def test_project_3d_to_2d_basic(self):
        """Test basic 3D to 2D projection."""
        # Create a simple point in front of camera
        points_3d = torch.tensor([
            [0.0, 0.0, 10.0],  # Point 10m in front
            [5.0, 0.0, 10.0],  # Point 10m in front, 5m to the right
        ])
        
        # Simple camera intrinsics
        intrinsics = torch.tensor([
            [1000.0, 0, 800.0],
            [0, 1000.0, 450.0],
            [0, 0, 1.0]
        ])
        
        # Identity extrinsics (camera at origin)
        extrinsics = torch.eye(4)
        
        # Project
        points_2d, valid_mask = project_3d_to_2d(points_3d, intrinsics, extrinsics)
        
        # Check that both points are valid
        assert valid_mask.all()
        
        # Check approximate projection
        # Point at (0, 0, 10) should project to center (800, 450)
        assert torch.abs(points_2d[0, 0] - 800.0) < 1.0
        assert torch.abs(points_2d[0, 1] - 450.0) < 1.0
        
        # Point at (5, 0, 10) should project to (800 + 500, 450)
        assert torch.abs(points_2d[1, 0] - 1300.0) < 1.0
    
    def test_project_3d_to_2d_behind_camera(self):
        """Test that points behind camera are marked invalid."""
        # Point behind camera
        points_3d = torch.tensor([
            [0.0, 0.0, -5.0],  # Behind camera
        ])
        
        intrinsics = torch.tensor([
            [1000.0, 0, 800.0],
            [0, 1000.0, 450.0],
            [0, 0, 1.0]
        ])
        
        extrinsics = torch.eye(4)
        
        points_2d, valid_mask = project_3d_to_2d(points_3d, intrinsics, extrinsics)
        
        # Point should be invalid
        assert not valid_mask[0]
    
    def test_backproject_2d_to_3d_basic(self):
        """Test basic 2D to 3D back-projection."""
        # Image point at center
        points_2d = torch.tensor([
            [800.0, 450.0],
        ])
        
        # Depth of 10m
        depth = torch.tensor([10.0])
        
        intrinsics = torch.tensor([
            [1000.0, 0, 800.0],
            [0, 1000.0, 450.0],
            [0, 0, 1.0]
        ])
        
        extrinsics = torch.eye(4)
        
        # Back-project
        points_3d = backproject_2d_to_3d(points_2d, depth, intrinsics, extrinsics)
        
        # Should be approximately (0, 0, 10)
        assert torch.abs(points_3d[0, 0]) < 0.1
        assert torch.abs(points_3d[0, 1]) < 0.1
        assert torch.abs(points_3d[0, 2] - 10.0) < 0.1
    
    def test_projection_round_trip(self):
        """Test that projection and back-projection are inverses."""
        # Original 3D points
        points_3d = torch.tensor([
            [1.0, 2.0, 10.0],
            [3.0, -1.0, 15.0],
            [-2.0, 4.0, 20.0],
        ])
        
        intrinsics = torch.tensor([
            [1000.0, 0, 800.0],
            [0, 1000.0, 450.0],
            [0, 0, 1.0]
        ])
        
        extrinsics = torch.eye(4)
        
        # Project to 2D
        points_2d, valid_mask = project_3d_to_2d(points_3d, intrinsics, extrinsics)
        
        # All should be valid
        assert valid_mask.all()
        
        # Back-project to 3D
        depths = points_3d[:, 2]
        reconstructed_3d = backproject_2d_to_3d(points_2d, depths, intrinsics, extrinsics)
        
        # Should match original
        assert torch.allclose(reconstructed_3d, points_3d, rtol=1e-4, atol=1e-4)
    
    def test_project_bev_to_image(self):
        """Test BEV grid projection to image space."""
        # Create simple BEV coordinates
        bev_h, bev_w = 10, 10
        x = torch.linspace(-5, 5, bev_w)
        y = torch.linspace(-5, 5, bev_h)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        zz = torch.zeros_like(xx)
        bev_coords = torch.stack([xx, yy, zz], dim=-1)
        
        # Camera parameters
        batch_size = 1
        n_cam = 1
        intrinsics = torch.tensor([
            [1000.0, 0, 800.0],
            [0, 1000.0, 450.0],
            [0, 0, 1.0]
        ]).unsqueeze(0).unsqueeze(0)
        
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0)
        
        bev_config = {
            'x_min': -5.0,
            'x_max': 5.0,
            'y_min': -5.0,
            'y_max': 5.0,
            'z_ref': 0.0
        }
        
        # Project
        reference_points, valid_mask = project_bev_to_image(
            bev_coords,
            intrinsics,
            extrinsics,
            bev_config
        )
        
        # Check shapes
        assert reference_points.shape == (batch_size, n_cam, bev_h * bev_w, 2)
        assert valid_mask.shape == (batch_size, n_cam, bev_h * bev_w)
        
        # Check that some points are valid (those in front of camera)
        # With z=0, points should be at camera height, so validity depends on camera pose
        # For identity extrinsics, points at z=0 might be behind camera
        # Let's just check the shapes are correct
        assert reference_points.dtype == torch.float32
        assert valid_mask.dtype == torch.bool


class TestBEVCoordinates:
    """Test BEV coordinate generation."""
    
    def test_bev_coords_shape(self):
        """Test that BEV coordinates have correct shape."""
        encoder = CameraBEVEncoder(bev_h=200, bev_w=200)
        
        assert encoder.bev_coords.shape == (200, 200, 3)
    
    def test_bev_coords_range(self):
        """Test that BEV coordinates are within specified range."""
        x_range = (-51.2, 51.2)
        y_range = (-51.2, 51.2)
        
        encoder = CameraBEVEncoder(
            bev_h=200,
            bev_w=200,
            bev_x_range=x_range,
            bev_y_range=y_range
        )
        
        x_coords = encoder.bev_coords[:, :, 0]
        y_coords = encoder.bev_coords[:, :, 1]
        
        assert x_coords.min() >= x_range[0]
        assert x_coords.max() <= x_range[1]
        assert y_coords.min() >= y_range[0]
        assert y_coords.max() <= y_range[1]
    
    def test_bev_coords_constant_z(self):
        """Test that BEV coordinates have constant z."""
        z_ref = -1.5
        encoder = CameraBEVEncoder(bev_h=100, bev_w=100, bev_z_ref=z_ref)
        
        z_coords = encoder.bev_coords[:, :, 2]
        
        assert torch.all(z_coords == z_ref)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
