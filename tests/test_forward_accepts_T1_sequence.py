"""
Tests for BEVFusionModel.forward() accepting T=1 temporal sequences.

These tests verify that:
1. forward() handles inputs with shape (B, T=1, ...) correctly
2. forward() doesn't crash when T=1 and returns valid outputs
3. return_intermediate=True produces fused_bev when expected
"""

import pytest
import torch

pytest.importorskip("modules.bev_fusion_model")

from modules.bev_fusion_model import BEVFusionModel
from modules.data_structures import BEVGridConfig


class TestForwardAcceptsT1Sequence:
    """Tests for forward() handling T=1 temporal inputs."""

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
        model.eval()
        return model

    def test_forward_with_T1_lidar_and_camera(self, model):
        """Test forward() with T=1 temporal dimension in inputs."""
        B, T, N_cam, N_pts = 1, 1, 6, 35000

        # Inputs with temporal dimension T=1
        lidar_points = torch.randn(B, T, N_pts, 4)
        camera_images = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)

        # Add some valid points
        lidar_points[:, :, :100, :3] = torch.randn(B, T, 100, 3) * 10

        with torch.no_grad():
            output = model.forward(
                lidar_points=lidar_points,
                camera_images=camera_images,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics,
            )

        # Should not crash and return valid outputs
        assert 'cls_scores' in output
        assert 'bbox_preds' in output
        assert not torch.isnan(output['cls_scores']).any()

    def test_forward_T1_with_lidar_mask(self, model):
        """Test forward() with T=1 and explicit lidar_mask."""
        B, T, N_cam, N_pts = 1, 1, 6, 35000

        lidar_points = torch.randn(B, T, N_pts, 4)
        camera_images = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)

        # Explicit mask with shape (B, T, N)
        lidar_mask = torch.zeros(B, T, N_pts, dtype=torch.bool)
        lidar_mask[:, :, :100] = True  # Only first 100 points valid

        with torch.no_grad():
            output = model.forward(
                lidar_points=lidar_points,
                camera_images=camera_images,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics,
                lidar_mask=lidar_mask,
            )

        assert 'cls_scores' in output
        assert not torch.isnan(output['cls_scores']).any()

    def test_forward_T1_return_intermediate(self, model):
        """Test forward() with T=1 and return_intermediate=True."""
        B, T, N_cam, N_pts = 1, 1, 6, 1000

        lidar_points = torch.randn(B, T, N_pts, 4) * 10
        camera_images = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)

        with torch.no_grad():
            output = model.forward(
                lidar_points=lidar_points,
                camera_images=camera_images,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics,
                return_intermediate=True,
            )

        # Should have intermediate features
        assert 'intermediate' in output
        intermediate = output['intermediate']

        # Should have lidar_bev, camera_bev, fused_bev
        assert 'lidar_bev' in intermediate
        assert 'camera_bev' in intermediate
        assert 'fused_bev' in intermediate

    def test_forward_3d_inputs_still_work(self, model):
        """Test forward() with standard 3D inputs (no T dimension) still works."""
        B, N_cam, N_pts = 1, 6, 1000

        # Standard 3D inputs (no temporal dimension)
        lidar_points = torch.randn(B, N_pts, 4) * 10
        camera_images = torch.randn(B, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, N_cam, -1, -1)

        with torch.no_grad():
            output = model.forward(
                lidar_points=lidar_points,
                camera_images=camera_images,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics,
            )

        assert 'cls_scores' in output
        assert not torch.isnan(output['cls_scores']).any()


class TestForwardDoesNotCrashWithVariousInputs:
    """Stress tests to ensure forward() is robust."""

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
            use_mc_convrnn=False,
            num_classes=10
        )
        model.eval()
        return model

    @pytest.mark.parametrize("B,T,N", [
        (1, 1, 100),
        (1, 1, 35000),
        (2, 1, 1000),
    ])
    def test_various_batch_and_point_sizes(self, model, B, T, N):
        """Test forward() with various batch sizes and point counts."""
        N_cam = 6

        lidar_points = torch.randn(B, T, N, 4) * 10
        camera_images = torch.randn(B, T, N_cam, 3, 224, 400)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)
        extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, T, N_cam, -1, -1)

        with torch.no_grad():
            output = model.forward(
                lidar_points=lidar_points,
                camera_images=camera_images,
                camera_intrinsics=intrinsics,
                camera_extrinsics=extrinsics,
            )

        assert 'cls_scores' in output
        assert output['cls_scores'].shape[0] == B


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
