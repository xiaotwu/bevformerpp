"""
Regression tests for LiDAR visualization extraction.

These tests verify that:
1. extract_lidar_points_np() handles various tensor shapes correctly
2. Object arrays and wrong shapes are detected with helpful errors
3. debug_lidar_bev() and project_lidar_to_bev() don't crash on valid inputs
4. Mask is correctly applied to filter padding
"""

import pytest
import numpy as np
import torch

pytest.importorskip("modules.vis.lidar_bev")

from modules.vis.lidar_bev import (
    extract_lidar_points_np,
    validate_points_np,
    debug_lidar_bev,
    project_lidar_to_bev,
)


class TestExtractLidarPointsNpFromTensor:
    """Tests for extract_lidar_points_np with various tensor shapes."""

    def test_extract_from_BTN4_tensor(self):
        """Test extraction from (B, T, N, 4) tensor."""
        B, T, N = 1, 1, 100
        points = torch.zeros((B, T, N, 4))
        # Fill first 10 points with random xyz
        points[:, :, :10, :3] = torch.randn(B, T, 10, 3) * 10
        points[:, :, :10, 3] = 1.0  # intensity

        # Create mask
        mask = torch.zeros((B, T, N), dtype=torch.bool)
        mask[:, :, :10] = True

        pts_np = extract_lidar_points_np(points, lidar_mask=mask)

        assert pts_np.shape == (10, 4), f"Expected (10, 4), got {pts_np.shape}"
        assert pts_np.dtype == np.float32

    def test_extract_from_BN4_tensor(self):
        """Test extraction from (B, N, 4) tensor."""
        B, N = 2, 500
        points = torch.randn(B, N, 4) * 10

        pts_np = extract_lidar_points_np(points, take_b=0)

        assert pts_np.shape == (N, 4)
        assert pts_np.dtype == np.float32

    def test_extract_from_N4_tensor(self):
        """Test extraction from (N, 4) tensor."""
        N = 1000
        points = torch.randn(N, 4) * 10

        pts_np = extract_lidar_points_np(points)

        assert pts_np.shape == (N, 4)
        assert pts_np.dtype == np.float32

    def test_extract_from_TN4_tensor(self):
        """Test extraction from (T, N, 4) tensor - temporal only."""
        T, N = 5, 200
        points = torch.randn(T, N, 4) * 10

        pts_np = extract_lidar_points_np(points, take_t="last")

        assert pts_np.shape == (N, 4)
        assert pts_np.dtype == np.float32

    def test_extract_takes_last_timestep(self):
        """Test that take_t='last' uses last timestep."""
        B, T, N = 1, 3, 100
        points = torch.zeros(B, T, N, 4)
        # Only last timestep has data
        points[:, -1, :, 0] = 42.0

        pts_np = extract_lidar_points_np(points, take_t="last")

        assert (pts_np[:, 0] == 42.0).all(), "Should have used last timestep"

    def test_extract_takes_first_timestep(self):
        """Test that take_t='first' uses first timestep."""
        B, T, N = 1, 3, 100
        points = torch.zeros(B, T, N, 4)
        # Only first timestep has data
        points[:, 0, :, 0] = 99.0

        pts_np = extract_lidar_points_np(points, take_t="first")

        assert (pts_np[:, 0] == 99.0).all(), "Should have used first timestep"


class TestObjectArrayStackRecovery:
    """Tests for recovering from object arrays."""

    def test_object_array_stack_recovery(self):
        """Test that object array of shape (N, 1) where each element is length-4 is stacked."""
        N = 50
        # Build object array where each element is a length-4 array
        obj_arr = np.empty((N, 1), dtype=object)
        for i in range(N):
            obj_arr[i, 0] = np.array([float(i), float(i+1), float(i+2), 1.0])

        pts_np = extract_lidar_points_np(obj_arr)

        # Should have been stacked to (N, 4)
        assert pts_np.shape == (N, 4), f"Expected ({N}, 4), got {pts_np.shape}"
        assert pts_np.dtype == np.float32

    def test_object_array_flat(self):
        """Test that flat object array is stacked correctly."""
        N = 30
        # Create a TRUE flat object array where each element is an array
        # (not using list comprehension with dtype=object, which creates 2D object array)
        obj_arr = np.empty(N, dtype=object)
        for i in range(N):
            obj_arr[i] = np.array([float(i), 0.0, 0.0, 1.0])

        pts_np = extract_lidar_points_np(obj_arr)

        assert pts_np.shape == (N, 4), f"Expected ({N}, 4), got {pts_np.shape}"


class TestDebugLidarBevNoCrash:
    """Tests that debug_lidar_bev doesn't crash on valid inputs."""

    def test_debug_lidar_bev_with_BTN4_input(self):
        """Test debug_lidar_bev with (B, T, N, 4) input."""
        B, T, N = 1, 1, 35000
        points = torch.randn(B, T, N, 4) * 10

        diag = debug_lidar_bev(points)

        assert 'num_points_total' in diag
        assert 'num_points_valid' in diag
        assert 'num_points_in_range' in diag
        assert diag['num_points_total'] > 0

    def test_debug_lidar_bev_with_mask(self):
        """Test debug_lidar_bev with mask filtering."""
        B, T, N = 1, 1, 1000
        points = torch.randn(B, T, N, 4) * 10
        mask = torch.zeros(B, T, N, dtype=torch.bool)
        mask[:, :, :100] = True

        diag = debug_lidar_bev(points, lidar_mask=mask)

        assert diag['num_points_total'] <= 100, "Should be filtered by mask"

    def test_project_lidar_to_bev_with_BTN4_input(self):
        """Test project_lidar_to_bev with (B, T, N, 4) input."""
        B, T, N = 1, 1, 500
        points = torch.randn(B, T, N, 4) * 10

        density = project_lidar_to_bev(points)

        assert density.ndim == 2
        assert density.dtype == np.float32
        assert not np.isnan(density).any()

    def test_project_lidar_to_bev_with_dict_input(self):
        """Test project_lidar_to_bev with dict input."""
        batch = {
            'lidar_points': torch.randn(1, 1, 500, 4) * 10,
            'lidar_mask': torch.ones(1, 1, 500, dtype=torch.bool),
        }

        density = project_lidar_to_bev(batch)

        assert density.ndim == 2


class TestInvalidShapeRaisesHelpfully:
    """Tests for helpful error messages on invalid shapes."""

    def test_shape_N1_float_raises_helpful_error(self):
        """Test that (N, 1) float array raises helpful error."""
        points = np.random.randn(1000, 1).astype(np.float32)

        with pytest.raises(ValueError) as exc_info:
            validate_points_np(points, require_cols=3)

        error_msg = str(exc_info.value)
        assert "columns" in error_msg.lower() or "3" in error_msg
        assert "1" in error_msg  # Should mention the actual column count

    def test_shape_N1_explains_object_array_issue(self):
        """Test that error for C=1 mentions object array possibility."""
        points = np.random.randn(1000, 1).astype(np.float32)

        with pytest.raises(ValueError) as exc_info:
            validate_points_np(points, require_cols=3)

        error_msg = str(exc_info.value)
        assert "C=1" in error_msg or "object" in error_msg.lower()

    def test_1d_array_raises(self):
        """Test that 1D array raises helpful error."""
        points = np.random.randn(100)

        with pytest.raises(ValueError) as exc_info:
            extract_lidar_points_np(points)

        error_msg = str(exc_info.value)
        assert "1D" in error_msg or "2D" in error_msg


class TestMaskFiltering:
    """Tests for mask-based padding filtering."""

    def test_mask_filters_padding(self):
        """Test that mask correctly filters padding points."""
        B, T, N = 1, 1, 1000
        points = torch.randn(B, T, N, 4) * 10
        # Only first 200 points are valid
        mask = torch.zeros(B, T, N, dtype=torch.bool)
        mask[:, :, :200] = True

        pts_np = extract_lidar_points_np(points, lidar_mask=mask)

        assert pts_np.shape[0] == 200, f"Expected 200 points, got {pts_np.shape[0]}"

    def test_mask_from_dict(self):
        """Test that mask is automatically extracted from dict."""
        N_valid = 50
        batch = {
            'lidar_points': torch.randn(1, 1, 500, 4),
            'lidar_mask': torch.zeros(1, 1, 500, dtype=torch.bool),
        }
        batch['lidar_mask'][:, :, :N_valid] = True

        pts_np = extract_lidar_points_np(batch)

        assert pts_np.shape[0] == N_valid

    def test_mask_2d_shape_BN(self):
        """Test that 2D mask (B, N) works correctly."""
        B, N = 1, 500
        points = torch.randn(B, N, 4) * 10
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, :100] = True

        pts_np = extract_lidar_points_np(points, lidar_mask=mask)

        assert pts_np.shape[0] == 100


class TestTransposedArrayHandling:
    """Tests for handling transposed arrays."""

    def test_transposed_array_detected_and_fixed(self):
        """Test that (C, N) array is transposed to (N, C)."""
        C, N = 4, 1000
        points = np.random.randn(C, N).astype(np.float32)

        pts_np = validate_points_np(points, require_cols=3)

        assert pts_np.shape == (N, C), f"Expected ({N}, {C}), got {pts_np.shape}"


class TestVisualizationIntegration:
    """Integration tests for visualization functions."""

    def test_extraction_for_visualization_workflow(self):
        """Test the full extraction workflow for visualization."""
        # Simulate a batch from the dataloader
        batch = {
            'lidar_points': torch.randn(1, 1, 35000, 4),
            'lidar_mask': torch.zeros(1, 1, 35000, dtype=torch.float32),
        }
        # First 500 points are valid
        batch['lidar_mask'][:, :, :500] = 1.0

        # This is what the visualization code does
        pts_np = extract_lidar_points_np(
            batch['lidar_points'],
            lidar_mask=batch['lidar_mask'],
            take_t='last',
            take_b=0,
            require_cols=3
        )

        # Should work without IndexError
        x = pts_np[:, 0]
        y = pts_np[:, 1]
        z = pts_np[:, 2]

        assert len(x) == 500
        assert len(y) == 500
        assert len(z) == 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
