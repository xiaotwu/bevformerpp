"""
Regression tests for LiDAR mask shape validation.

These tests ensure that the exact bug that caused:
    "boolean index did not match indexed array along axis 1;
     size of axis is 35000 but size of corresponding boolean axis is 4"

is caught early with clear error messages and cannot recur.

The bug was caused by using .any(dim=1) instead of .any(dim=-1) when
inferring mask from points, which produced a mask of shape (B, 4) instead
of (B, N).
"""

import pytest
import torch
import numpy as np

# Import the helper functions
pytest.importorskip("modules.utils.core")
from modules.utils.core import (
    normalize_lidar_points_and_mask,
    validate_lidar_mask_shape,
)


class TestMaskShapeValidation:
    """Tests for mask shape validation catching the (B, 4) bug."""

    def test_wrong_mask_shape_B4_raises_error(self):
        """Test that a mask with shape (B, 4) raises ValueError with helpful message."""
        B, N = 2, 35000
        points = torch.randn(B, N, 4)
        wrong_mask = torch.ones(B, 4, dtype=torch.bool)  # WRONG: should be (B, N)

        with pytest.raises(ValueError) as exc_info:
            normalize_lidar_points_and_mask(points, wrong_mask)

        # Error message should mention the shape mismatch
        error_msg = str(exc_info.value)
        assert "4" in error_msg or "shape" in error_msg.lower()

    def test_wrong_mask_shape_BN4_raises_error(self):
        """Test that a mask with shape (B, N, 4) raises ValueError."""
        B, N = 2, 1000
        points = torch.randn(B, N, 4)
        wrong_mask = torch.ones(B, N, 4, dtype=torch.bool)  # WRONG: 3D mask

        with pytest.raises(ValueError) as exc_info:
            normalize_lidar_points_and_mask(points, wrong_mask)

        error_msg = str(exc_info.value)
        assert "dimension" in error_msg.lower() or "shape" in error_msg.lower()

    def test_correct_mask_shape_BN_works(self):
        """Test that correct mask shape (B, N) works without error."""
        B, N = 2, 35000
        points = torch.randn(B, N, 4)
        correct_mask = torch.ones(B, N, dtype=torch.bool)

        # Should not raise
        pts, msk = normalize_lidar_points_and_mask(points, correct_mask)

        assert pts.shape == (B, N, 4)
        assert msk.shape == (B, N)
        assert msk.dtype == torch.bool

    def test_validate_lidar_mask_shape_catches_B4_bug(self):
        """Test that validate_lidar_mask_shape catches (B, 4) masks."""
        wrong_mask = torch.ones(2, 4, dtype=torch.bool)

        with pytest.raises(ValueError) as exc_info:
            validate_lidar_mask_shape(wrong_mask, expected_batch=2, expected_points=35000)

        error_msg = str(exc_info.value)
        # Error should mention the dimension reduction bug
        assert "4" in error_msg
        assert "35000" in error_msg or "expected" in error_msg.lower()

    def test_validate_lidar_mask_shape_provides_guidance(self):
        """Test that error message provides guidance about .any(dim=1) vs .any(dim=-1)."""
        wrong_mask = torch.ones(2, 4, dtype=torch.bool)

        with pytest.raises(ValueError) as exc_info:
            validate_lidar_mask_shape(wrong_mask, expected_batch=2, expected_points=35000)

        error_msg = str(exc_info.value)
        # Should mention the common error pattern
        assert "dim=1" in error_msg or "dim=-1" in error_msg


class TestMaskInference:
    """Tests for correct mask inference from points."""

    def test_mask_inferred_with_correct_dimension(self):
        """Test that inferred mask has correct shape (B, N) not (B, 4)."""
        B, N = 2, 35000
        points = torch.randn(B, N, 4)
        # Make some points zero (padding)
        points[:, N // 2:, :] = 0.0

        pts, msk = normalize_lidar_points_and_mask(points, mask=None)

        # Mask should be (B, N), not (B, 4)!
        assert msk.shape == (B, N), f"Mask has wrong shape {msk.shape}, expected {(B, N)}"
        assert msk.dtype == torch.bool

        # First half should be valid, second half should be invalid (padding)
        for b in range(B):
            assert msk[b, :N // 2].all(), "Non-zero points should be marked valid"
            assert not msk[b, N // 2:].any(), "Zero points should be marked invalid"

    def test_mask_inference_uses_sum_dim_minus1(self):
        """Test that mask inference correctly uses reduction over last dimension."""
        B, N = 1, 1000
        points = torch.zeros(B, N, 4)

        # Set only first 100 points to have non-zero xyz
        points[0, :100, :3] = torch.randn(100, 3)

        pts, msk = normalize_lidar_points_and_mask(points, mask=None)

        # Exactly 100 points should be marked as valid
        assert msk[0, :100].all()
        assert not msk[0, 100:].any()
        assert msk.sum() == 100


class TestTemporalDimensionHandling:
    """Tests for handling 4D temporal inputs (B, T, N, 4)."""

    def test_4d_input_T1_squeezes_correctly(self):
        """Test that 4D input with T=1 is squeezed to 3D."""
        B, T, N = 2, 1, 35000
        points_4d = torch.randn(B, T, N, 4)

        pts, msk = normalize_lidar_points_and_mask(points_4d)

        assert pts.shape == (B, N, 4)
        assert msk.shape == (B, N)

    def test_4d_input_with_3d_mask_squeezes_correctly(self):
        """Test that 4D input with 3D mask (B, T, N) is handled."""
        B, T, N = 2, 1, 1000
        points_4d = torch.randn(B, T, N, 4)
        mask_3d = torch.ones(B, T, N, dtype=torch.bool)

        pts, msk = normalize_lidar_points_and_mask(points_4d, mask_3d)

        assert pts.shape == (B, N, 4)
        assert msk.shape == (B, N)

    def test_4d_input_multiframe_uses_last_timestep(self):
        """Test that 4D input with T>1 uses last timestep."""
        B, T, N = 2, 5, 1000
        points_4d = torch.randn(B, T, N, 4)
        # Make last timestep distinctive
        points_4d[:, -1, :, 0] = 999.0

        pts, msk = normalize_lidar_points_and_mask(points_4d, squeeze_temporal=True)

        # Should have used last timestep
        assert pts.shape == (B, N, 4)
        assert (pts[:, :, 0] == 999.0).all()


class TestEncoderIntegration:
    """Integration tests with LiDARBEVEncoder."""

    @pytest.fixture
    def encoder(self):
        pytest.importorskip("modules.lidar_encoder")
        from modules.lidar_encoder import LiDARBEVEncoder
        from modules.data_structures import BEVGridConfig

        config = BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)
        encoder = LiDARBEVEncoder(config=config, out_channels=64)
        encoder.eval()
        return encoder

    def test_encoder_rejects_B4_mask(self, encoder):
        """Test that encoder's forward_padded rejects (B, 4) mask."""
        B, N = 1, 35000
        points = torch.randn(B, N, 4)
        wrong_mask = torch.ones(B, 4, dtype=torch.bool)  # WRONG shape

        with pytest.raises(ValueError):
            encoder.forward_padded(points, mask=wrong_mask)

    def test_encoder_accepts_4d_points(self, encoder):
        """Test that encoder handles 4D input (B, T, N, 4)."""
        B, T, N = 1, 1, 35000
        points_4d = torch.randn(B, T, N, 4)
        # Add some valid points
        points_4d[:, :, :100, :3] = torch.randn(B, T, 100, 3) * 10

        with torch.no_grad():
            output = encoder.forward_padded(points_4d)

        assert output.shape[0] == B
        assert not torch.isnan(output).any()

    def test_encoder_handles_matching_4d_mask(self, encoder):
        """Test that encoder handles matching 4D points and 3D mask."""
        B, T, N = 1, 1, 1000
        points_4d = torch.randn(B, T, N, 4) * 10
        mask_3d = torch.ones(B, T, N, dtype=torch.bool)
        mask_3d[:, :, 500:] = False  # Mask out second half

        with torch.no_grad():
            output = encoder.forward_padded(points_4d, mask=mask_3d)

        assert output.shape[0] == B
        assert not torch.isnan(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
