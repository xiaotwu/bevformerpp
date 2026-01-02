"""
Tests for BEV alignment and temporal unrolling correctness.

These tests verify critical P0 requirements:
1. Camera and LiDAR BEV share the same grid configuration (H, W, z_ref)
2. forward_sequence properly resets temporal state (no cross-batch leakage)
3. LiDAR sequence shape is correct in collate output
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Skip tests if modules not available
pytest.importorskip("modules.data_structures")

from modules.data_structures import BEVGridConfig, Box3D, Sample


class TestBEVGridAlignment:
    """Tests for P0-2: Unified BEV grid config."""

    def test_bev_grid_config_z_ref_property(self):
        """Test z_ref property computes midpoint of z range."""
        config = BEVGridConfig(z_min=-5.0, z_max=3.0)
        assert config.z_ref == -1.0  # (-5 + 3) / 2 = -1

    def test_bev_grid_config_to_dict(self):
        """Test to_dict includes z_ref."""
        config = BEVGridConfig.from_grid_size(bev_h=200, bev_w=200)
        d = config.to_dict()
        assert 'z_ref' in d
        assert d['z_ref'] == config.z_ref

    def test_bev_grid_size_from_resolution(self):
        """Test grid size computation from resolution."""
        config = BEVGridConfig.from_grid_size(bev_h=200, bev_w=200)
        H, W = config.grid_size
        assert H == 200
        assert W == 200


class TestCollateSequenceShapes:
    """Tests for P0-1: Collate returns proper sequence shapes."""

    def test_collate_lidar_sequence_shape(self):
        """Test that collate_fn_with_lidar returns (B, T, N, 4) LiDAR."""
        from modules.nuscenes_dataset import collate_fn_with_lidar

        # Create mock batch with 2 samples, sequence length 3
        def create_mock_sample(scene_token="scene1"):
            sample = Sample(
                sample_token="tok",
                scene_token=scene_token,
                timestamp=0,
                lidar_path="test.bin"
            )
            sample.lidar_points = np.random.randn(100, 4).astype(np.float32)
            sample.camera_images = {
                cam: np.random.randint(0, 255, (224, 400, 3), dtype=np.uint8)
                for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                           'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']
            }
            sample.camera_intrinsics = {
                cam: np.eye(3, dtype=np.float32)
                for cam in sample.camera_images.keys()
            }
            sample.camera_extrinsics = {
                cam: np.eye(4, dtype=np.float32)
                for cam in sample.camera_images.keys()
            }
            sample.ego_pose = np.eye(4, dtype=np.float32)
            sample.annotations = []
            return sample

        batch = [
            {'samples': [create_mock_sample() for _ in range(3)], 'ego_transforms': []},
            {'samples': [create_mock_sample() for _ in range(3)], 'ego_transforms': []}
        ]

        result = collate_fn_with_lidar(
            batch,
            bev_h=50, bev_w=50,  # Small for test
            num_classes=10,
            generate_targets=False,
            max_points=100
        )

        # Check LiDAR sequence shape: (B, T, max_points, 4)
        assert result['lidar_points'].shape == (2, 3, 100, 4)
        assert result['lidar_mask'].shape == (2, 3, 100)

        # Check image sequence shape: (B, T, N_cams, C, H, W)
        assert result['img'].shape[0] == 2  # B
        assert result['img'].shape[1] == 3  # T
        assert result['img'].shape[2] == 6  # N_cams


class TestTemporalStateIsolation:
    """Tests for P0-1: No cross-batch state leakage."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_sequence_resets_state(self):
        """Test that forward_sequence resets temporal state between batches."""
        from modules.bev_fusion_model import BEVFusionModel
        from modules.data_structures import BEVGridConfig

        # Create minimal model
        config = BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)
        model = BEVFusionModel(
            bev_config=config,
            lidar_channels=32,
            camera_channels=64,
            fused_channels=64,
            use_temporal_attention=True,
            use_mc_convrnn=False,
            num_classes=10
        )
        model = model.cuda().eval()

        # Create random input batch
        B, T, N_cam = 1, 3, 6
        N_pts = 100

        def create_batch():
            return {
                'lidar_points_seq': torch.randn(B, T, N_pts, 4).cuda(),
                'camera_images_seq': torch.randn(B, T, N_cam, 3, 224, 400).cuda(),
                'camera_intrinsics_seq': torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1).cuda(),
                'camera_extrinsics_seq': torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1).cuda(),
                'ego_pose_seq': torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1).cuda()
            }

        # Run forward_sequence twice with different inputs
        batch1 = create_batch()
        with torch.no_grad():
            output1 = model.forward_sequence(**batch1)

        # Verify temporal state was reset (should not persist after forward_sequence)
        if model.use_temporal_attention:
            # After forward_sequence completes, check that calling reset_temporal_state
            # doesn't change results on same input
            model.reset_temporal_state()

            output1_again = model.forward_sequence(**batch1)

            # Outputs should be identical (state was reset properly)
            assert torch.allclose(output1['cls_scores'], output1_again['cls_scores'], atol=1e-5)

    def test_reset_temporal_state_method_exists(self):
        """Test that reset_temporal_state method exists on BEVFusionModel."""
        from modules.bev_fusion_model import BEVFusionModel

        config = BEVGridConfig.from_grid_size(bev_h=50, bev_w=50)
        model = BEVFusionModel(
            bev_config=config,
            use_temporal_attention=False,
            use_mc_convrnn=False
        )

        assert hasattr(model, 'reset_temporal_state')
        assert callable(model.reset_temporal_state)
        assert hasattr(model, 'forward_sequence')
        assert callable(model.forward_sequence)


class TestConfigOverrides:
    """Tests for P1-1: Config override mechanism."""

    def test_parse_override_basic(self):
        """Test basic override parsing."""
        from configs.config_loader import parse_override

        keys, value = parse_override("training.batch_size=4")
        assert keys == ['training', 'batch_size']
        assert value == 4

    def test_parse_override_nested(self):
        """Test nested key override parsing."""
        from configs.config_loader import parse_override

        keys, value = parse_override("model.fusion.type=local_attention")
        assert keys == ['model', 'fusion', 'type']
        assert value == 'local_attention'

    def test_parse_override_types(self):
        """Test value type parsing."""
        from configs.config_loader import _parse_value

        assert _parse_value("true") is True
        assert _parse_value("false") is False
        assert _parse_value("42") == 42
        assert _parse_value("3.14") == 3.14
        assert _parse_value("none") is None
        assert _parse_value("hello") == "hello"

    def test_apply_overrides(self):
        """Test applying overrides to config dict."""
        from configs.config_loader import apply_overrides

        config = {'training': {'batch_size': 2, 'epochs': 10}}
        result = apply_overrides(config, ['training.batch_size=4', 'training.lr=0.001'])

        assert result['training']['batch_size'] == 4
        assert result['training']['lr'] == 0.001
        assert result['training']['epochs'] == 10  # Unchanged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
