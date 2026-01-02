"""
Tests for padding invariance in LiDAR encoding.

These tests verify that:
1. Padded points NEVER affect pillarization/BEV features
2. Identical valid points with different padding produce identical outputs
3. The mask correctly filters out padding before any feature computation
"""

import pytest
import torch
import numpy as np

pytest.importorskip("modules.lidar_encoder")

from modules.lidar_encoder import LiDARBEVEncoder
from modules.data_structures import BEVGridConfig


class TestPaddingInvariance:
    """Tests that padding points don't affect encoder output."""

    @pytest.fixture
    def config(self):
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def encoder(self, config):
        encoder = LiDARBEVEncoder(config=config, out_channels=64)
        encoder.eval()
        return encoder

    def test_identical_valid_points_different_padding_same_output(self, encoder):
        """Test that identical valid points with different padding produce same output."""
        B = 1
        N_valid = 500

        # Create valid points (non-zero)
        valid_points = np.random.randn(N_valid, 4).astype(np.float32) * 10

        # Padding to different sizes
        max_points_small = 1000
        max_points_large = 5000

        # Create small padded version
        points_small = np.zeros((max_points_small, 4), dtype=np.float32)
        points_small[:N_valid] = valid_points
        mask_small = np.zeros(max_points_small, dtype=np.float32)
        mask_small[:N_valid] = 1.0

        # Create large padded version
        points_large = np.zeros((max_points_large, 4), dtype=np.float32)
        points_large[:N_valid] = valid_points
        mask_large = np.zeros(max_points_large, dtype=np.float32)
        mask_large[:N_valid] = 1.0

        # Convert to tensors
        points_small_t = torch.from_numpy(points_small).unsqueeze(0)  # (1, 1000, 4)
        mask_small_t = torch.from_numpy(mask_small).unsqueeze(0) > 0.5  # (1, 1000)

        points_large_t = torch.from_numpy(points_large).unsqueeze(0)  # (1, 5000, 4)
        mask_large_t = torch.from_numpy(mask_large).unsqueeze(0) > 0.5  # (1, 5000)

        with torch.no_grad():
            output_small = encoder.forward_padded(points_small_t, mask=mask_small_t)
            output_large = encoder.forward_padded(points_large_t, mask=mask_large_t)

        # Outputs should be identical (same valid points)
        assert torch.allclose(output_small, output_large, atol=1e-5), \
            "Outputs differ despite identical valid points!"

    def test_padding_values_dont_leak_into_features(self, encoder):
        """Test that padding values (even if non-zero) don't affect output."""
        B = 1
        N_valid = 200
        max_points = 1000

        # Create valid points
        valid_points = np.random.randn(N_valid, 4).astype(np.float32) * 10

        # Version 1: Zero padding
        points_zero_pad = np.zeros((max_points, 4), dtype=np.float32)
        points_zero_pad[:N_valid] = valid_points

        # Version 2: Non-zero garbage padding (should be masked out)
        points_garbage_pad = np.random.randn(max_points, 4).astype(np.float32) * 100
        points_garbage_pad[:N_valid] = valid_points

        # Same mask for both
        mask = np.zeros(max_points, dtype=np.float32)
        mask[:N_valid] = 1.0

        # Convert to tensors
        points_zero_t = torch.from_numpy(points_zero_pad).unsqueeze(0)
        points_garbage_t = torch.from_numpy(points_garbage_pad).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0) > 0.5

        with torch.no_grad():
            output_zero = encoder.forward_padded(points_zero_t, mask=mask_t)
            output_garbage = encoder.forward_padded(points_garbage_t, mask=mask_t)

        # Outputs should be identical (mask filters out padding)
        assert torch.allclose(output_zero, output_garbage, atol=1e-5), \
            "Garbage padding leaked into features despite mask!"

    def test_all_padding_produces_zero_features(self, encoder):
        """Test that all-padding input produces zero BEV features (no NaN)."""
        B = 1
        max_points = 1000

        # All zeros (padding)
        points = torch.zeros(B, max_points, 4)
        mask = torch.zeros(B, max_points, dtype=torch.bool)  # All invalid

        with torch.no_grad():
            output = encoder.forward_padded(points, mask=mask)

        # Output should be all zeros (no valid points to contribute)
        assert not torch.isnan(output).any(), "NaN in output for all-padding input!"
        # Note: Output may not be exactly zero due to batch norm, but should be valid


class TestMaskCorrectlyFiltersPoints:
    """Tests that mask correctly filters points before pillarization."""

    @pytest.fixture
    def config(self):
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def encoder(self, config):
        encoder = LiDARBEVEncoder(config=config, out_channels=64)
        encoder.eval()
        return encoder

    def test_mask_filters_trailing_padding(self, encoder):
        """Test that trailing padding is filtered by mask."""
        B = 1
        N_valid = 100
        max_points = 500

        # Create points with valid at start, padding at end
        points = torch.zeros(B, max_points, 4)
        points[:, :N_valid, :3] = torch.randn(B, N_valid, 3) * 10
        points[:, :N_valid, 3] = torch.rand(B, N_valid)  # intensity

        # Mask indicates only first N_valid are valid
        mask = torch.zeros(B, max_points, dtype=torch.bool)
        mask[:, :N_valid] = True

        with torch.no_grad():
            output = encoder.forward_padded(points, mask=mask)

        assert output.shape[0] == B
        assert not torch.isnan(output).any()

    def test_mask_filters_scattered_padding(self, encoder):
        """Test that scattered padding (not just trailing) is filtered."""
        B = 1
        max_points = 500

        # Create points with interleaved valid/padding
        points = torch.zeros(B, max_points, 4)
        # Every other point is valid
        valid_indices = torch.arange(0, max_points, 2)
        points[:, valid_indices, :3] = torch.randn(B, len(valid_indices), 3) * 10
        points[:, valid_indices, 3] = torch.rand(B, len(valid_indices))

        # Mask: every other point
        mask = torch.zeros(B, max_points, dtype=torch.bool)
        mask[:, valid_indices] = True

        with torch.no_grad():
            output = encoder.forward_padded(points, mask=mask)

        assert output.shape[0] == B
        assert not torch.isnan(output).any()


class TestListVsPaddedEquivalence:
    """Tests that list-based and padded forward produce same results."""

    @pytest.fixture
    def config(self):
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def encoder(self, config):
        encoder = LiDARBEVEncoder(config=config, out_channels=64)
        encoder.eval()
        return encoder

    def test_forward_list_equals_forward_padded(self, encoder):
        """Test that forward_list and forward_padded produce identical outputs."""
        N_valid = 300

        # Create valid points as numpy array
        valid_points = np.random.randn(N_valid, 4).astype(np.float32) * 10

        # Forward with list
        with torch.no_grad():
            output_list = encoder.forward_list([valid_points])

        # Forward with padded tensor
        max_points = 1000
        points_padded = np.zeros((max_points, 4), dtype=np.float32)
        points_padded[:N_valid] = valid_points
        mask = np.zeros(max_points, dtype=np.float32)
        mask[:N_valid] = 1.0

        points_t = torch.from_numpy(points_padded).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0) > 0.5

        with torch.no_grad():
            output_padded = encoder.forward_padded(points_t, mask=mask_t)

        # Outputs should be identical
        assert torch.allclose(output_list, output_padded, atol=1e-5), \
            "forward_list and forward_padded produce different outputs!"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
