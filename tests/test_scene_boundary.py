"""
Tests for temporal scene-boundary resets (requirement 2.4).

These tests verify that:
1. Scene-boundary detection correctly identifies when scene tokens change
2. Temporal state is reset for the affected sample when a scene boundary is detected
3. Other samples in the batch are not affected by one sample's scene change
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Skip if modules not available
pytest.importorskip("modules.bev_fusion_model")

from modules.bev_fusion_model import BEVFusionModel
from modules.data_structures import BEVGridConfig


class TestSceneBoundaryDetection:
    """Tests for scene-boundary detection in forward_sequence."""

    @pytest.fixture
    def config(self):
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def model(self, config):
        model = BEVFusionModel(
            bev_config=config,
            lidar_channels=32,
            camera_channels=64,
            fused_channels=64,
            use_temporal_attention=False,
            use_mc_convrnn=True,  # Use MC-ConvRNN to test state reset
            num_classes=10
        )
        model.eval()
        return model

    def test_scene_tokens_none_no_error(self, model):
        """Test that scene_tokens=None doesn't cause errors."""
        B, T, N_cam, N_pts = 1, 3, 6, 100

        # Create dummy inputs
        lidar = torch.randn(B, T, N_pts, 4)
        imgs = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        ego_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        with torch.no_grad():
            output = model.forward_sequence(
                lidar_points_seq=lidar,
                camera_images_seq=imgs,
                camera_intrinsics_seq=intrinsics,
                camera_extrinsics_seq=extrinsics,
                ego_pose_seq=ego_pose,
                scene_tokens=None  # Should not cause error
            )

        assert 'cls_scores' in output
        assert not torch.isnan(output['cls_scores']).any()

    def test_same_scene_no_reset(self, model):
        """Test that same scene tokens don't trigger reset."""
        B, T, N_cam, N_pts = 1, 3, 6, 100

        lidar = torch.randn(B, T, N_pts, 4)
        imgs = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        ego_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Same scene token for all timesteps
        scene_tokens = [['scene_1'] for _ in range(T)]

        # Track if reset was called
        reset_called = []
        original_reset = model._reset_temporal_state_sample

        def mock_reset(sample_idx):
            reset_called.append(sample_idx)
            original_reset(sample_idx)

        model._reset_temporal_state_sample = mock_reset

        with torch.no_grad():
            output = model.forward_sequence(
                lidar_points_seq=lidar,
                camera_images_seq=imgs,
                camera_intrinsics_seq=intrinsics,
                camera_extrinsics_seq=extrinsics,
                ego_pose_seq=ego_pose,
                scene_tokens=scene_tokens
            )

        # Reset should not be called (same scene throughout)
        assert len(reset_called) == 0, f"Reset called unexpectedly: {reset_called}"

    def test_scene_boundary_triggers_reset(self, model):
        """Test that scene boundary triggers temporal state reset."""
        B, T, N_cam, N_pts = 1, 3, 6, 100

        lidar = torch.randn(B, T, N_pts, 4)
        imgs = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        ego_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Scene changes at t=2
        scene_tokens = [
            ['scene_1'],  # t=0
            ['scene_1'],  # t=1
            ['scene_2'],  # t=2 - scene boundary!
        ]

        # Track if reset was called
        reset_called = []
        original_reset = model._reset_temporal_state_sample

        def mock_reset(sample_idx):
            reset_called.append(sample_idx)
            original_reset(sample_idx)

        model._reset_temporal_state_sample = mock_reset

        with torch.no_grad():
            output = model.forward_sequence(
                lidar_points_seq=lidar,
                camera_images_seq=imgs,
                camera_intrinsics_seq=intrinsics,
                camera_extrinsics_seq=extrinsics,
                ego_pose_seq=ego_pose,
                scene_tokens=scene_tokens
            )

        # Reset should be called once for sample 0 (scene changed at t=2)
        assert len(reset_called) == 1, f"Expected 1 reset call, got {len(reset_called)}"
        assert reset_called[0] == 0, f"Reset called for wrong sample: {reset_called[0]}"

    def test_batch_scene_boundary_isolation(self, model):
        """Test that scene boundary in one sample doesn't affect others."""
        B, T, N_cam, N_pts = 2, 3, 6, 100

        lidar = torch.randn(B, T, N_pts, 4)
        imgs = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        ego_pose = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)

        # Sample 0: scene changes at t=2
        # Sample 1: same scene throughout
        scene_tokens = [
            ['scene_A', 'scene_B'],  # t=0
            ['scene_A', 'scene_B'],  # t=1
            ['scene_A_2', 'scene_B'],  # t=2 - only sample 0 has boundary
        ]

        # Track if reset was called
        reset_called = []
        original_reset = model._reset_temporal_state_sample

        def mock_reset(sample_idx):
            reset_called.append(sample_idx)
            original_reset(sample_idx)

        model._reset_temporal_state_sample = mock_reset

        with torch.no_grad():
            output = model.forward_sequence(
                lidar_points_seq=lidar,
                camera_images_seq=imgs,
                camera_intrinsics_seq=intrinsics,
                camera_extrinsics_seq=extrinsics,
                ego_pose_seq=ego_pose,
                scene_tokens=scene_tokens
            )

        # Reset should only be called for sample 0
        assert len(reset_called) == 1, f"Expected 1 reset call, got {len(reset_called)}"
        assert reset_called[0] == 0, f"Reset called for wrong sample: {reset_called[0]}"


class TestPerSampleStateReset:
    """Tests for per-sample temporal state reset."""

    @pytest.fixture
    def config(self):
        return BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)

    @pytest.fixture
    def model(self, config):
        model = BEVFusionModel(
            bev_config=config,
            lidar_channels=32,
            camera_channels=64,
            fused_channels=64,
            use_temporal_attention=False,
            use_mc_convrnn=True,
            num_classes=10
        )
        return model

    def test_reset_temporal_state_sample_zeros_hidden(self, model):
        """Test that _reset_temporal_state_sample zeros out hidden state."""
        B = 2

        # Initialize fake hidden state
        model.mc_hidden_state = torch.ones(B, 64, 50, 50)

        # Reset sample 0
        model._reset_temporal_state_sample(0)

        # Sample 0 should be zeros
        assert (model.mc_hidden_state[0] == 0).all()
        # Sample 1 should still be ones
        assert (model.mc_hidden_state[1] == 1).all()

    def test_reset_sample_method_exists(self, model):
        """Test that reset method exists."""
        assert hasattr(model, '_reset_temporal_state_sample')
        assert callable(model._reset_temporal_state_sample)

    def test_full_reset_zeros_all_samples(self, model):
        """Test that full reset_temporal_state zeros all samples."""
        B = 2

        # Initialize fake hidden state
        model.mc_hidden_state = torch.ones(B, 64, 50, 50)

        # Full reset
        model.reset_temporal_state()

        # Should be None now
        assert model.mc_hidden_state is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
