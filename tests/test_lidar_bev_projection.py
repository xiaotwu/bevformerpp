"""
Tests for LiDAR BEV projection utilities.

Verifies that:
1. Point cloud projection handles padding correctly
2. Density map shows proper spatial distribution (not a single dot)
3. Coordinate conventions are correct
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.vis.lidar_bev import (
    project_lidar_to_bev,
    debug_lidar_bev,
)


class TestProjectLidarToBev:
    """Tests for the project_lidar_to_bev function."""
    
    def test_basic_projection(self):
        """Test basic projection with known points."""
        # Create a simple point cloud with 4 points in each quadrant
        points = np.array([
            [10.0, 10.0, 0.0, 1.0],   # Quadrant 1 (+x, +y)
            [-10.0, 10.0, 0.0, 1.0],  # Quadrant 2 (-x, +y)
            [-10.0, -10.0, 0.0, 1.0], # Quadrant 3 (-x, -y)
            [10.0, -10.0, 0.0, 1.0],  # Quadrant 4 (+x, -y)
        ], dtype=np.float32)
        
        density = project_lidar_to_bev(
            points,
            x_range=(-20.0, 20.0),
            y_range=(-20.0, 20.0),
            resolution=0.5,
        )
        
        # Should have H=80, W=80
        assert density.shape == (80, 80), f"Expected (80, 80), got {density.shape}"
        
        # Should have exactly 4 non-zero cells
        assert density.sum() == 4, f"Expected 4 points, got {density.sum()}"
        
        # Non-zero cells should be spread out (not in single location)
        nonzero_y, nonzero_x = np.where(density > 0)
        assert len(nonzero_y) == 4, f"Expected 4 non-zero cells, got {len(nonzero_y)}"
        
        # Check that points are in different quadrants of the grid
        center = 40  # Grid center
        quadrants = set()
        for y, x in zip(nonzero_y, nonzero_x):
            q = (y >= center, x >= center)
            quadrants.add(q)
        
        assert len(quadrants) == 4, "Points should be in all 4 quadrants"
    
    def test_padding_filtered(self):
        """Test that zero-padded points are filtered out."""
        # Create points with padding (zeros)
        points = np.zeros((35000, 4), dtype=np.float32)
        points[:10] = np.array([
            [5.0, 5.0, 0.0, 1.0],
            [5.1, 5.1, 0.0, 1.0],
            [-5.0, 5.0, 0.0, 1.0],
            [-5.0, -5.0, 0.0, 1.0],
            [5.0, -5.0, 0.0, 1.0],
            [0.1, 0.1, 0.0, 1.0],  # Near origin but not exactly zero
            [0.2, 0.2, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 0.0, 1.0],
            [3.0, 3.0, 0.0, 1.0],
        ])
        
        density = project_lidar_to_bev(
            points,
            x_range=(-10.0, 10.0),
            y_range=(-10.0, 10.0),
            resolution=0.5,
        )
        
        # Should only count the 10 valid points, not 35000
        assert density.sum() == 10, f"Expected 10, got {density.sum()} (padding not filtered)"
    
    def test_torch_tensor_input(self):
        """Test that torch.Tensor input works correctly."""
        points_np = np.array([
            [5.0, 5.0, 0.0, 1.0],
            [-5.0, 5.0, 0.0, 1.0],
            [-5.0, -5.0, 0.0, 1.0],
            [5.0, -5.0, 0.0, 1.0],
        ], dtype=np.float32)
        
        points_torch = torch.from_numpy(points_np)
        
        density_np = project_lidar_to_bev(points_np, x_range=(-10, 10), y_range=(-10, 10), resolution=0.5)
        density_torch = project_lidar_to_bev(points_torch, x_range=(-10, 10), y_range=(-10, 10), resolution=0.5)
        
        np.testing.assert_array_equal(density_np, density_torch)
    
    def test_z_filtering(self):
        """Test z-range filtering."""
        # Note: (0,0,0,*) is filtered as padding, so avoid origin
        points = np.array([
            [0.1, 0.1, 0.0, 1.0],   # Within z range
            [1.0, 0.0, 0.0, 1.0],   # Within z range
            [2.0, 0.0, 10.0, 1.0],  # Above z range
            [3.0, 0.0, -10.0, 1.0], # Below z range
        ], dtype=np.float32)
        
        density = project_lidar_to_bev(
            points,
            x_range=(-5, 5),
            y_range=(-5, 5),
            resolution=0.5,
            z_range=(-2.0, 2.0),
        )
        
        # Only 2 points should be within z range
        assert density.sum() == 2, f"Expected 2 points after z filter, got {density.sum()}"


class TestDebugLidarBev:
    """Tests for diagnostic metrics."""
    
    def test_fraction_clipped_sanity(self):
        """
        Verify fraction_clipped is NOT ~1.0 for typical nuScenes-like data.
        
        This catches the bug where most/all points are outside the BEV range.
        """
        # Simulate nuScenes-like point cloud distribution
        # Most points should be within the standard 51.2m range
        np.random.seed(42)
        n_points = 10000
        
        # Realistic distribution: most points within 50m, some outliers
        r = np.random.exponential(scale=20, size=n_points)  # Distance from ego
        theta = np.random.uniform(0, 2*np.pi, n_points)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(-2, 2, n_points)
        intensity = np.random.uniform(0, 1, n_points)
        
        points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
        
        diag = debug_lidar_bev(
            points,
            x_range=(-51.2, 51.2),
            y_range=(-51.2, 51.2),
            z_range=(-5.0, 3.0),
        )
        
        # Most points should be in range (fraction_clipped < 0.95)
        assert diag['fraction_clipped'] < 0.95, \
            f"Too many points clipped: {diag['fraction_clipped']:.1%}. " \
            f"This suggests BEV projection is wrong."
        
        # Density should not be concentrated in single pixel
        # top1 / sum should be < 0.5
        if diag['density_sum'] > 0:
            top1_ratio = diag['density_max'] / diag['density_sum']
            assert top1_ratio < 0.5, \
                f"Density too concentrated: top1/sum = {top1_ratio:.2f}. " \
                f"Expected distributed density, got single peak."
    
    def test_padding_detection(self):
        """Test that padding is correctly detected."""
        # Create heavily padded data (like batched loader output)
        points = np.zeros((35000, 4), dtype=np.float32)
        points[:100] = np.random.randn(100, 4).astype(np.float32) * 10
        
        diag = debug_lidar_bev(points)
        
        assert diag['num_points_total'] == 35000
        assert diag['num_points_valid'] <= 100 + 1  # +1 for potential (0,0,0) real point


class TestCoordinateConventions:
    """Tests for coordinate system correctness."""
    
    def test_x_axis_is_columns(self):
        """Verify that x-axis maps to columns (width)."""
        # Point at positive x only
        points = np.array([[10.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        
        density = project_lidar_to_bev(
            points,
            x_range=(-20, 20),
            y_range=(-20, 20),
            resolution=1.0,
        )
        
        # Grid is 40x40, point at x=10 should be at column 30 (20 + 10)
        nonzero_y, nonzero_x = np.where(density > 0)
        
        assert len(nonzero_x) == 1
        # x=10 with x_range=(-20,20), res=1.0 -> col = (10 - (-20)) / 1.0 = 30
        assert nonzero_x[0] == 30, f"Expected column 30, got {nonzero_x[0]}"
        # y=0 -> row = (0 - (-20)) / 1.0 = 20
        assert nonzero_y[0] == 20, f"Expected row 20, got {nonzero_y[0]}"
    
    def test_y_axis_is_rows(self):
        """Verify that y-axis maps to rows (height)."""
        # Point at positive y only
        points = np.array([[0.0, 10.0, 0.0, 1.0]], dtype=np.float32)
        
        density = project_lidar_to_bev(
            points,
            x_range=(-20, 20),
            y_range=(-20, 20),
            resolution=1.0,
        )
        
        nonzero_y, nonzero_x = np.where(density > 0)
        
        assert len(nonzero_y) == 1
        # y=10 with y_range=(-20,20), res=1.0 -> row = (10 - (-20)) / 1.0 = 30
        assert nonzero_y[0] == 30, f"Expected row 30, got {nonzero_y[0]}"
        # x=0 -> col = 20
        assert nonzero_x[0] == 20, f"Expected column 20, got {nonzero_x[0]}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

