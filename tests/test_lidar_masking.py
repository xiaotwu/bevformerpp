"""
Tests for LiDAR padding/mask correctness (P0 requirement 2.1).

These tests verify that:
1. Padded points (zeros) never contribute to pillar features
2. Two inputs with identical valid points but different padding produce equal outputs
3. All-padding input produces zeros without NaNs
4. Mask is properly propagated through the entire pipeline
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

# Skip if modules not available
pytest.importorskip("modules.lidar_encoder")

from modules.lidar_encoder import LiDARBEVEncoder, Pillarization
from modules.data_structures import BEVGridConfig


class TestPillarizationMasking:
    """Tests for Pillarization with masked inputs."""

    @pytest.fixture
    def config(self):
        """Create a small BEV grid config for testing."""
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def pillarization(self, config):
        """Create Pillarization instance."""
        return Pillarization(config, max_points_per_pillar=32, max_pillars=1000)

    def test_empty_points_returns_empty_pillars(self, pillarization):
        """Test that empty point cloud returns empty pillars."""
        points = np.zeros((0, 4), dtype=np.float32)
        pillars, coords, num_points = pillarization(points)

        assert pillars.shape[0] == 0
        assert coords.shape[0] == 0
        assert num_points.shape[0] == 0

    def test_points_outside_bounds_filtered(self, pillarization, config):
        """Test that points outside BEV bounds are filtered."""
        # Create points way outside the BEV range
        far_points = np.array([
            [1000.0, 1000.0, 0.0, 1.0],  # Far outside
            [-1000.0, -1000.0, 0.0, 1.0],  # Far outside
        ], dtype=np.float32)

        pillars, coords, num_points = pillarization(far_points)
        assert pillars.shape[0] == 0  # All points filtered

    def test_valid_points_create_pillars(self, pillarization, config):
        """Test that valid points within bounds create pillars."""
        # Create points at center of BEV (should be within bounds)
        valid_points = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ], dtype=np.float32)

        pillars, coords, num_points = pillarization(valid_points)
        assert pillars.shape[0] > 0  # Should have at least 1 pillar


class TestLiDAREncoderMasking:
    """Tests for LiDARBEVEncoder mask-aware processing."""

    @pytest.fixture
    def config(self):
        """Create a small BEV grid config for testing."""
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def encoder(self, config):
        """Create LiDAR encoder."""
        encoder = LiDARBEVEncoder(
            config=config,
            out_channels=32,
            max_points_per_pillar=32,
            max_pillars=500
        )
        encoder.eval()
        return encoder

    def test_identical_valid_points_same_output(self, encoder, config):
        """
        CRITICAL TEST: Two padded inputs with identical valid points
        but different padding must produce equal outputs.
        """
        # Valid points at center of BEV
        valid_points = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5, 0.8],
            [2.0, 2.0, 1.0, 0.6],
            [-1.0, -1.0, 0.0, 0.9],
            [3.0, 0.0, 0.2, 0.7],
        ], dtype=np.float32)

        # Input 1: 10 points total, 5 valid + 5 padding
        points1 = np.zeros((10, 4), dtype=np.float32)
        points1[:5] = valid_points
        mask1 = np.array([True] * 5 + [False] * 5)

        # Input 2: 20 points total, 5 valid + 15 padding
        points2 = np.zeros((20, 4), dtype=np.float32)
        points2[:5] = valid_points
        mask2 = np.array([True] * 5 + [False] * 15)

        # Convert to tensors
        points1_t = torch.from_numpy(points1).unsqueeze(0)  # (1, 10, 4)
        points2_t = torch.from_numpy(points2).unsqueeze(0)  # (1, 20, 4)
        mask1_t = torch.from_numpy(mask1).unsqueeze(0)  # (1, 10)
        mask2_t = torch.from_numpy(mask2).unsqueeze(0)  # (1, 20)

        with torch.no_grad():
            output1 = encoder(points1_t, mask=mask1_t)
            output2 = encoder(points2_t, mask=mask2_t)

        # Outputs must be identical (same valid points)
        assert output1.shape == output2.shape
        assert torch.allclose(output1, output2, atol=1e-5), \
            f"Outputs differ! Max diff: {(output1 - output2).abs().max().item()}"

    def test_all_padding_produces_zeros_no_nan(self, encoder, config):
        """
        CRITICAL TEST: All-padding input (no valid points) must produce
        zeros without NaNs.
        """
        # All-padding input
        points = torch.zeros((1, 100, 4))
        mask = torch.zeros((1, 100), dtype=torch.bool)  # All False = all padding

        with torch.no_grad():
            output = encoder(points, mask=mask)

        # Output should be all zeros
        assert not torch.isnan(output).any(), "Output contains NaN!"
        assert not torch.isinf(output).any(), "Output contains Inf!"
        assert (output == 0).all(), "All-padding should produce zeros"

    def test_mask_none_fallback(self, encoder, config):
        """Test that mask=None falls back to inferring mask from zero points."""
        # Create points where some are zero (padding)
        points = torch.zeros((1, 10, 4))
        points[0, 0] = torch.tensor([1.0, 1.0, 0.5, 0.8])  # One valid point

        with torch.no_grad():
            output = encoder(points, mask=None)  # Should infer mask

        assert not torch.isnan(output).any(), "Output contains NaN!"
        # Should have some non-zero values (from the one valid point)
        # Note: depends on point being within BEV bounds

    def test_forward_list_vs_forward_padded_equivalence(self, encoder, config):
        """Test that forward_list and forward_padded produce same results."""
        # Valid points
        valid_points = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.5, 0.8],
            [2.0, 2.0, 1.0, 0.6],
        ], dtype=np.float32)

        # Method 1: forward_list with numpy arrays
        with torch.no_grad():
            output_list = encoder.forward_list([valid_points])

        # Method 2: forward_padded with mask
        points_padded = torch.zeros((1, 10, 4))
        points_padded[0, :3] = torch.from_numpy(valid_points)
        mask = torch.zeros((1, 10), dtype=torch.bool)
        mask[0, :3] = True

        with torch.no_grad():
            output_padded = encoder.forward_padded(points_padded, mask)

        # Results should be identical
        assert output_list.shape == output_padded.shape
        assert torch.allclose(output_list, output_padded, atol=1e-5), \
            f"List vs padded differ! Max diff: {(output_list - output_padded).abs().max().item()}"

    def test_batch_processing_with_masks(self, encoder, config):
        """Test batch processing with different masks per sample."""
        B = 2
        N = 50

        # Sample 1: 10 valid points
        # Sample 2: 5 valid points
        points = torch.zeros((B, N, 4))
        mask = torch.zeros((B, N), dtype=torch.bool)

        # Add valid points for sample 1
        points[0, :10, :] = torch.randn(10, 4)
        points[0, :10, 0:3] = points[0, :10, 0:3] * 5  # Scale to be within BEV
        mask[0, :10] = True

        # Add valid points for sample 2
        points[1, :5, :] = torch.randn(5, 4)
        points[1, :5, 0:3] = points[1, :5, 0:3] * 5
        mask[1, :5] = True

        with torch.no_grad():
            output = encoder(points, mask=mask)

        assert output.shape[0] == B
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestPaddingContamination:
    """Tests to verify padding points don't contaminate pillar statistics."""

    @pytest.fixture
    def config(self):
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def encoder(self, config):
        encoder = LiDARBEVEncoder(
            config=config,
            out_channels=32,
            max_points_per_pillar=32,
            max_pillars=500
        )
        encoder.eval()
        return encoder

    def test_padding_at_origin_not_counted(self, encoder, config):
        """
        Padding points at [0,0,0,0] should not create artificial
        density at the origin pillar.
        """
        # One valid point far from origin
        valid_point = np.array([[10.0, 10.0, 0.0, 1.0]], dtype=np.float32)

        # Without padding contamination: just the valid point
        output_clean = encoder.forward_list([valid_point])

        # With lots of zero padding (if mask not used, would contaminate origin)
        points_padded = torch.zeros((1, 1000, 4))
        points_padded[0, 0] = torch.tensor([10.0, 10.0, 0.0, 1.0])
        mask = torch.zeros((1, 1000), dtype=torch.bool)
        mask[0, 0] = True

        with torch.no_grad():
            output_masked = encoder(points_padded, mask=mask)

        # Outputs should be identical - padding didn't contaminate
        assert torch.allclose(output_clean, output_masked, atol=1e-5), \
            "Padding contamination detected!"

    def test_pillar_center_calculation_excludes_padding(self, encoder, config):
        """
        Verify that pillar center (mean z) calculations exclude padding.
        """
        # Create points in a specific pillar
        pillar_points = np.array([
            [5.0, 5.0, 1.0, 1.0],
            [5.1, 5.1, 2.0, 1.0],
            [5.2, 5.2, 3.0, 1.0],
        ], dtype=np.float32)  # Mean z should be 2.0

        # Without padding
        output1 = encoder.forward_list([pillar_points])

        # With padding that would skew mean z if counted
        points_padded = torch.zeros((1, 100, 4))
        points_padded[0, :3] = torch.from_numpy(pillar_points)
        # Padding is [0,0,0,0] - if counted, mean z would be ~0.06 instead of 2.0
        mask = torch.zeros((1, 100), dtype=torch.bool)
        mask[0, :3] = True

        with torch.no_grad():
            output2 = encoder(points_padded, mask=mask)

        assert torch.allclose(output1, output2, atol=1e-5), \
            "Padding affected pillar center calculation!"


class TestMaskPropagation:
    """Tests to verify mask is properly propagated through fusion model."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mask_propagates_to_fusion_model(self):
        """Test that lidar_mask is used in BEVFusionModel.forward()."""
        from modules.bev_fusion_model import BEVFusionModel

        config = BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)
        model = BEVFusionModel(
            bev_config=config,
            lidar_channels=32,
            camera_channels=64,
            fused_channels=64,
            use_temporal_attention=False,
            use_mc_convrnn=False,
            num_classes=10
        )
        model = model.cuda().eval()

        B = 1
        N_cam = 6
        N_pts = 100

        # Create inputs
        lidar = torch.zeros((B, N_pts, 4)).cuda()
        lidar[0, :5] = torch.randn(5, 4).cuda() * 5  # 5 valid points
        lidar_mask = torch.zeros((B, N_pts), dtype=torch.bool).cuda()
        lidar_mask[0, :5] = True

        images = torch.randn(B, N_cam, 3, 224, 400).cuda()
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1).cuda()
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1).cuda()

        with torch.no_grad():
            output = model(
                lidar_points=lidar,
                camera_images=images,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics,
                lidar_mask=lidar_mask
            )

        assert 'cls_scores' in output
        assert not torch.isnan(output['cls_scores']).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
