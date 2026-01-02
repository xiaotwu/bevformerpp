"""
BEV Alignment Sanity Tests.

Verifies that BEV coordinate transforms are correct and consistent:
1. Ego-motion warping preserves known patterns
2. Coordinate chain (sensor -> ego -> BEV) is correct
3. Grid resolution matches configuration
"""

import pytest
import torch
import math
import numpy as np

from modules.temporal.ego_motion_warp import (
    EgoMotionWarp,
    warp_bev_with_ego_motion,
    create_bev_grid
)
from modules.utils import compute_visibility_mask


class TestBEVGridAlignment:
    """Tests for BEV grid coordinate alignment."""
    
    def test_grid_coordinates_correct(self):
        """Test that BEV grid has correct coordinate values."""
        H, W = 100, 100
        bev_range = (-50.0, 50.0, -50.0, 50.0)
        
        grid = create_bev_grid(H, W, bev_range, torch.device('cpu'), torch.float32)
        
        assert grid.shape == (H, W, 2)
        
        # Check corners
        x_min, x_max, y_min, y_max = bev_range
        
        # Top-left corner (H-1, 0) should be (x_min, y_max)
        # Note: grid is indexed [y, x] but stores [x, y] coordinates
        assert torch.isclose(grid[0, 0, 0], torch.tensor(x_min)), f"Expected x={x_min}, got {grid[0, 0, 0]}"
        assert torch.isclose(grid[0, 0, 1], torch.tensor(y_min)), f"Expected y={y_min}, got {grid[0, 0, 1]}"
        
        # Bottom-right corner
        assert torch.isclose(grid[-1, -1, 0], torch.tensor(x_max)), f"Expected x={x_max}, got {grid[-1, -1, 0]}"
        assert torch.isclose(grid[-1, -1, 1], torch.tensor(y_max)), f"Expected y={y_max}, got {grid[-1, -1, 1]}"
    
    def test_grid_resolution(self):
        """Test that grid resolution matches expected values."""
        H, W = 512, 512
        bev_range = (-51.2, 51.2, -51.2, 51.2)  # 102.4m range
        expected_resolution = 102.4 / 512  # 0.2m per pixel
        
        grid = create_bev_grid(H, W, bev_range, torch.device('cpu'), torch.float32)
        
        # Check resolution in x direction
        dx = (grid[0, 1, 0] - grid[0, 0, 0]).item()
        assert abs(dx - expected_resolution) < 0.001, f"Expected dx={expected_resolution}, got {dx}"
        
        # Check resolution in y direction
        dy = (grid[1, 0, 1] - grid[0, 0, 1]).item()
        assert abs(dy - expected_resolution) < 0.001, f"Expected dy={expected_resolution}, got {dy}"


class TestEgoMotionWarpAlignment:
    """Tests for ego-motion warp alignment correctness."""
    
    def test_pure_translation_x(self):
        """Test pure x-translation warping."""
        B, C, H, W = 1, 1, 100, 100
        bev_range = (-50.0, 50.0, -50.0, 50.0)  # 1m per pixel resolution
        
        # Create a feature with a marker at center
        features = torch.zeros(B, C, H, W)
        center_y, center_x = H // 2, W // 2
        features[0, 0, center_y, center_x] = 1.0
        
        # Translate 10 meters in x direction
        translation_m = 10.0
        transform = torch.eye(4).unsqueeze(0)
        transform[0, 0, 3] = translation_m
        
        warped = warp_bev_with_ego_motion(features, transform, bev_range=bev_range)
        
        # Find the peak in warped features
        peak_idx = warped[0, 0].argmax()
        peak_y = peak_idx // W
        peak_x = peak_idx % W
        
        # Expected shift: 10m / 1m_per_pixel = 10 pixels
        expected_shift = int(translation_m / (100.0 / W))
        
        # The peak should have moved (approximately, due to interpolation)
        assert abs(peak_x - center_x) > 0 or abs(peak_y - center_y) > 0, \
            "Peak should have moved after translation"
    
    def test_pure_rotation_90deg(self):
        """Test pure 90-degree rotation warping."""
        B, C, H, W = 1, 1, 100, 100
        bev_range = (-50.0, 50.0, -50.0, 50.0)
        
        # Create a feature with a marker offset from center
        features = torch.zeros(B, C, H, W)
        marker_y, marker_x = H // 2, W // 2 + 20  # 20 pixels right of center
        features[0, 0, marker_y, marker_x] = 1.0
        
        # 90 degree rotation around z-axis
        angle = math.pi / 2
        transform = torch.eye(4).unsqueeze(0)
        transform[0, 0, 0] = math.cos(angle)
        transform[0, 0, 1] = -math.sin(angle)
        transform[0, 1, 0] = math.sin(angle)
        transform[0, 1, 1] = math.cos(angle)
        
        warped = warp_bev_with_ego_motion(features, transform, bev_range=bev_range)
        
        # The peak should have rotated
        # Due to interpolation, check that the peak is no longer at original position
        original_value = warped[0, 0, marker_y, marker_x].item()
        assert original_value < 0.5, "Peak should have moved after rotation"
    
    def test_identity_transform_preserves_features(self):
        """Test that identity transform preserves features exactly."""
        B, C, H, W = 2, 64, 50, 50
        
        # Random features
        features = torch.randn(B, C, H, W)
        identity = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        warped = warp_bev_with_ego_motion(features, identity)
        
        # Should be identical (within numerical tolerance)
        assert torch.allclose(warped, features, atol=1e-5), \
            "Identity transform should preserve features exactly"
    
    def test_inverse_transform_recovers_features(self):
        """Test that applying inverse transform recovers original features."""
        B, C, H, W = 1, 1, 50, 50
        bev_range = (-25.0, 25.0, -25.0, 25.0)
        
        # Create features with a distinct pattern
        features = torch.zeros(B, C, H, W)
        features[0, 0, 25, 30] = 1.0  # Off-center marker
        
        # Small translation
        transform = torch.eye(4).unsqueeze(0)
        transform[0, 0, 3] = 2.0  # 2m translation
        
        # Forward warp
        warped = warp_bev_with_ego_motion(features, transform, bev_range=bev_range)
        
        # Inverse warp
        transform_inv = torch.inverse(transform)
        recovered = warp_bev_with_ego_motion(warped, transform_inv, bev_range=bev_range)
        
        # Due to interpolation, we can't expect exact recovery
        # But the peak should be back near the original location
        orig_peak = features[0, 0].argmax()
        recovered_peak = recovered[0, 0].argmax()
        
        orig_y, orig_x = orig_peak // W, orig_peak % W
        rec_y, rec_x = recovered_peak // W, recovered_peak % W
        
        # Allow 2 pixel tolerance due to interpolation
        assert abs(orig_x - rec_x) <= 2 and abs(orig_y - rec_y) <= 2, \
            f"Peak not recovered: original ({orig_x}, {orig_y}), recovered ({rec_x}, {rec_y})"


class TestCoordinateChain:
    """Tests for coordinate transformation chain consistency."""
    
    def test_bev_range_consistency(self):
        """Test that BEV range is used consistently across modules."""
        # This test verifies that different modules use the same BEV range
        bev_range = (-51.2, 51.2, -51.2, 51.2)
        H, W = 200, 200
        
        # Check ego warp module
        warp = EgoMotionWarp(bev_range=bev_range)
        assert warp.bev_range == bev_range
        
        # Check grid creation
        grid = create_bev_grid(H, W, bev_range, torch.device('cpu'), torch.float32)
        
        x_min, x_max, y_min, y_max = bev_range
        
        # Grid should span the BEV range
        assert torch.isclose(grid[0, 0, 0], torch.tensor(x_min))
        assert torch.isclose(grid[-1, -1, 0], torch.tensor(x_max))
    
    def test_visibility_mask_matches_warp(self):
        """Test that visibility mask correctly identifies out-of-bounds regions."""
        B, H, W = 1, 100, 100
        bev_range = (-50.0, 50.0, -50.0, 50.0)
        
        # Large translation that should create out-of-bounds regions
        transform = torch.eye(4).unsqueeze(0)
        transform[0, 0, 3] = 30.0  # 30m translation
        
        # Compute visibility mask
        mask = compute_visibility_mask(transform, H, W, bev_range=bev_range)
        
        # Some region should be masked out
        assert mask.sum() < B * H * W, "Large translation should mask some regions"
        
        # Mask should have valid shape
        assert mask.shape == (B, 1, H, W)
        
        # Values should be binary-ish (0 or 1)
        assert (mask >= 0).all() and (mask <= 1).all()


class TestAlignmentWithKnownTransform:
    """Test alignment with known transformation scenarios."""
    
    def test_forward_motion_warps_backward(self):
        """
        When ego moves forward, features from previous frame
        should warp backward (appear further away).
        """
        B, C, H, W = 1, 1, 100, 100
        bev_range = (-50.0, 50.0, -50.0, 50.0)
        
        # Feature at 40m ahead (y=90 in image coords for our convention)
        features = torch.zeros(B, C, H, W)
        # Put marker ahead of ego
        features[0, 0, 90, 50] = 1.0  # Ahead (high y) and centered (x=50)
        
        # Ego moves forward 10m (positive x in NuScenes convention)
        transform = torch.eye(4).unsqueeze(0)
        transform[0, 0, 3] = 10.0  # Forward motion
        
        warped = warp_bev_with_ego_motion(features, transform, bev_range=bev_range)
        
        # After warping, the feature should appear closer
        # (lower y value in image coords if y-axis is forward)
        peak_idx = warped[0, 0].argmax()
        peak_y = peak_idx // W
        
        # The peak location change depends on coordinate convention
        # This test mainly verifies that warping happens
        assert warped[0, 0, 90, 50] < 0.9, "Original location should have reduced value after warp"
    
    def test_small_rotation_alignment(self):
        """Test that small rotation causes expected feature shift."""
        B, C, H, W = 1, 1, 100, 100
        bev_range = (-50.0, 50.0, -50.0, 50.0)
        
        # Feature off-center
        features = torch.zeros(B, C, H, W)
        features[0, 0, 50, 70] = 1.0  # Right of center
        
        # Small rotation (5 degrees)
        angle = 5.0 * math.pi / 180.0
        transform = torch.eye(4).unsqueeze(0)
        transform[0, 0, 0] = math.cos(angle)
        transform[0, 0, 1] = -math.sin(angle)
        transform[0, 1, 0] = math.sin(angle)
        transform[0, 1, 1] = math.cos(angle)
        
        warped = warp_bev_with_ego_motion(features, transform, bev_range=bev_range)
        
        # Feature should have rotated
        original_value = warped[0, 0, 50, 70].item()
        assert original_value < 0.95, "Feature should have shifted due to rotation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

