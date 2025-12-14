"""
Unit tests for dataset loading functionality.
Tests sample loading completeness, calibration matrix shapes, and ego-motion computation.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from modules.nuscenes_dataset import NuScenesLoader, NuScenesDataset
from modules.data_structures import Sample, Box3D, BEVGridConfig
from modules.preprocessing import PointCloudPreprocessor, ImagePreprocessor, BEVTransform


class TestNuScenesLoader:
    """Test NuScenesLoader functionality."""
    
    @pytest.fixture
    def loader(self):
        """Create a NuScenesLoader instance."""
        return NuScenesLoader(dataroot="data", version="v1.0-mini")
    
    def test_loader_initialization(self, loader):
        """Test that loader initializes correctly."""
        assert loader.dataroot == Path("data")
        assert loader.version == "v1.0-mini"
        assert len(loader.tables) > 0
        assert 'sample' in loader.tables
        assert 'scene' in loader.tables
    
    def test_tables_loaded(self, loader):
        """Test that all required tables are loaded."""
        required_tables = [
            'attribute', 'calibrated_sensor', 'category', 'ego_pose',
            'instance', 'log', 'map', 'sample', 'sample_annotation',
            'sample_data', 'scene', 'sensor', 'visibility'
        ]
        for table in required_tables:
            assert table in loader.tables
            assert len(loader.tables[table]) > 0
    
    def test_get_record(self, loader):
        """Test getting a record by token."""
        # Get first sample
        sample = loader.tables['sample'][0]
        token = sample['token']
        
        # Retrieve using get method
        retrieved = loader.get('sample', token)
        assert retrieved['token'] == token
        assert retrieved == sample


class TestNuScenesDataset:
    """Test NuScenesDataset functionality."""
    
    @pytest.fixture
    def dataset(self):
        """Create a NuScenesDataset instance."""
        return NuScenesDataset(
            dataroot="data",
            version="v1.0-mini",
            split="train",
            sequence_length=1,
            load_lidar=True,
            load_images=True,
            load_annotations=True
        )
    
    def test_dataset_initialization(self, dataset):
        """Test that dataset initializes correctly."""
        assert len(dataset) > 0
        assert dataset.sequence_length == 1
        assert dataset.load_lidar is True
        assert dataset.load_images is True
        assert dataset.load_annotations is True
    
    def test_sample_loading_completeness(self, dataset):
        """
        Test sample loading completeness.
        Validates: Requirements 7.2
        """
        # Get first sample
        data = dataset[0]
        
        assert 'samples' in data
        assert 'ego_transforms' in data
        
        samples = data['samples']
        assert len(samples) == dataset.sequence_length
        
        # Check first sample
        sample = samples[0]
        assert isinstance(sample, Sample)
        
        # Check LiDAR data
        assert sample.lidar_path is not None
        assert sample.lidar_points is not None
        assert sample.lidar_points.shape[1] == 4  # x, y, z, intensity
        assert sample.lidar_points.shape[0] > 0  # Has points
        
        # Check camera data - should have 6 cameras
        assert len(sample.camera_paths) == 6
        assert len(sample.camera_images) == 6
        assert len(sample.camera_intrinsics) == 6
        assert len(sample.camera_extrinsics) == 6
        
        # Check camera names
        expected_cameras = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
        ]
        for cam_name in expected_cameras:
            assert cam_name in sample.camera_paths
            assert cam_name in sample.camera_images
            assert cam_name in sample.camera_intrinsics
            assert cam_name in sample.camera_extrinsics
        
        # Check ego pose
        assert sample.ego_pose is not None
        assert sample.ego_pose.shape == (4, 4)
        
        # Check annotations
        assert isinstance(sample.annotations, list)
        # Annotations may be empty for some samples, so just check type
    
    def test_calibration_matrix_shapes(self, dataset):
        """
        Test calibration matrix shapes.
        Validates: Requirements 7.3
        """
        data = dataset[0]
        sample = data['samples'][0]
        
        # Test intrinsics shape
        for cam_name, intrinsic in sample.camera_intrinsics.items():
            assert intrinsic.shape == (3, 3), \
                f"Intrinsic matrix for {cam_name} has wrong shape: {intrinsic.shape}"
            
            # Check that it's a valid camera matrix
            assert intrinsic[2, 2] == 1.0, "Bottom-right element should be 1"
            assert intrinsic[0, 0] > 0, "Focal length fx should be positive"
            assert intrinsic[1, 1] > 0, "Focal length fy should be positive"
        
        # Test extrinsics shape
        for cam_name, extrinsic in sample.camera_extrinsics.items():
            assert extrinsic.shape == (4, 4), \
                f"Extrinsic matrix for {cam_name} has wrong shape: {extrinsic.shape}"
            
            # Check that it's a valid SE(3) matrix
            assert np.allclose(extrinsic[3, :], [0, 0, 0, 1]), \
                "Bottom row should be [0, 0, 0, 1]"
            
            # Check rotation part is orthogonal
            R = extrinsic[:3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), \
                f"Rotation matrix for {cam_name} is not orthogonal"
    
    def test_ego_motion_computation(self, dataset):
        """
        Test ego-motion computation between frames.
        Validates: Requirements 7.4
        """
        # Create dataset with sequence length > 1
        dataset_seq = NuScenesDataset(
            dataroot="data",
            version="v1.0-mini",
            split="train",
            sequence_length=3,
            load_lidar=False,
            load_images=False,
            load_annotations=False
        )
        
        data = dataset_seq[0]
        samples = data['samples']
        ego_transforms = data['ego_transforms']
        
        # Should have sequence_length - 1 transforms
        assert len(ego_transforms) == len(samples) - 1
        
        # Check each transform
        for i, transform in enumerate(ego_transforms):
            assert transform.shape == (4, 4), \
                f"Ego transform {i} has wrong shape: {transform.shape}"
            
            # Check that it's a valid SE(3) matrix
            assert np.allclose(transform[3, :], [0, 0, 0, 1]), \
                "Bottom row should be [0, 0, 0, 1]"
            
            # Check rotation part is orthogonal
            R = transform[:3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-5), \
                f"Rotation matrix for transform {i} is not orthogonal"
            
            # Verify transform is computed correctly
            # transform should map from frame i to frame i+1
            ego_i = samples[i].ego_pose
            ego_next = samples[i + 1].ego_pose
            
            # Expected: inv(ego_next) @ ego_i
            expected_transform = np.linalg.inv(ego_next) @ ego_i
            
            assert np.allclose(transform, expected_transform, atol=1e-5), \
                f"Ego transform {i} does not match expected computation"
    
    def test_box3d_creation(self, dataset):
        """Test Box3D dataclass creation and validation."""
        # Create a valid box
        box = Box3D(
            center=np.array([1.0, 2.0, 0.5]),
            size=np.array([2.0, 4.0, 1.5]),
            yaw=0.5,
            label='car',
            score=0.9,
            token='test_token'
        )
        
        assert box.center.shape == (3,)
        assert box.size.shape == (3,)
        assert box.yaw == 0.5
        assert box.label == 'car'
        assert box.score == 0.9
        
        # Test invalid box (negative size)
        with pytest.raises(AssertionError):
            Box3D(
                center=np.array([0, 0, 0]),
                size=np.array([-1, 2, 1]),  # Negative size
                yaw=0,
                label='car'
            )
        
        # Test invalid score
        with pytest.raises(AssertionError):
            Box3D(
                center=np.array([0, 0, 0]),
                size=np.array([1, 2, 1]),
                yaw=0,
                label='car',
                score=1.5  # Score > 1
            )


class TestPointCloudPreprocessor:
    """Test point cloud preprocessing utilities."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a PointCloudPreprocessor instance."""
        config = BEVGridConfig()
        return PointCloudPreprocessor(config)
    
    def test_filter_points(self, preprocessor):
        """Test point cloud filtering."""
        # Create test points (some in range, some out)
        points = np.array([
            [0, 0, 0, 100],      # In range
            [100, 0, 0, 100],    # Out of x range
            [0, 100, 0, 100],    # Out of y range
            [0, 0, 10, 100],     # Out of z range
            [10, 10, 1, 100],    # In range
        ], dtype=np.float32)
        
        filtered = preprocessor.filter_points(points)
        
        # Should keep only points 0 and 4
        assert filtered.shape[0] == 2
        assert np.allclose(filtered[0], points[0])
        assert np.allclose(filtered[1], points[4])
    
    def test_points_to_bev_coords(self, preprocessor):
        """Test conversion to BEV coordinates."""
        # Create test points
        points = np.array([
            [0, 0, 0, 100],      # Center of BEV
            [51.2, 51.2, 0, 100],  # Max corner
            [-51.2, -51.2, 0, 100],  # Min corner
        ], dtype=np.float32)
        
        bev_coords = preprocessor.points_to_bev_coords(points)
        
        assert bev_coords.shape == (3, 2)
        
        # Center should map to middle of grid
        H, W = preprocessor.bev_config.grid_size
        assert bev_coords[0, 0] == W // 2
        assert bev_coords[0, 1] == H // 2


class TestImagePreprocessor:
    """Test image preprocessing utilities."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create an ImagePreprocessor instance."""
        return ImagePreprocessor(target_size=(256, 704))
    
    def test_resize(self, preprocessor):
        """Test image resizing."""
        # Create test image
        image = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
        
        resized = preprocessor.resize(image)
        
        assert resized.shape == (256, 704, 3)
    
    def test_normalize(self, preprocessor):
        """Test image normalization."""
        # Create test image
        image = np.random.randint(0, 255, (256, 704, 3), dtype=np.uint8)
        
        normalized = preprocessor.normalize(image)
        
        assert normalized.shape == image.shape
        assert normalized.dtype == np.float32
        # Values should be roughly in [-2, 2] range after normalization
        assert normalized.min() >= -3
        assert normalized.max() <= 3
    
    def test_adjust_intrinsics(self, preprocessor):
        """Test intrinsics adjustment for resizing."""
        # Original intrinsics for 1600x900 image
        intrinsics = np.array([
            [1000, 0, 800],
            [0, 1000, 450],
            [0, 0, 1]
        ], dtype=np.float32)
        
        adjusted = preprocessor.adjust_intrinsics(intrinsics, (900, 1600))
        
        # Check that focal lengths and principal point are scaled
        scale_x = 704 / 1600
        scale_y = 256 / 900
        
        assert np.isclose(adjusted[0, 0], intrinsics[0, 0] * scale_x)
        assert np.isclose(adjusted[1, 1], intrinsics[1, 1] * scale_y)
        assert np.isclose(adjusted[0, 2], intrinsics[0, 2] * scale_x)
        assert np.isclose(adjusted[1, 2], intrinsics[1, 2] * scale_y)


class TestBEVTransform:
    """Test BEV coordinate transformation utilities."""
    
    @pytest.fixture
    def transform(self):
        """Create a BEVTransform instance."""
        config = BEVGridConfig()
        return BEVTransform(config)
    
    def test_world_to_bev(self, transform):
        """Test world to BEV coordinate transformation."""
        # Test center point
        points = np.array([[0, 0, 0]], dtype=np.float32)
        bev_coords = transform.world_to_bev(points)
        
        H, W = transform.bev_config.grid_size
        assert bev_coords[0, 0] == W // 2
        assert bev_coords[0, 1] == H // 2
    
    def test_bev_to_world(self, transform):
        """Test BEV to world coordinate transformation."""
        H, W = transform.bev_config.grid_size
        
        # Test center pixel
        coords = np.array([[W // 2, H // 2]], dtype=np.int32)
        world_coords = transform.bev_to_world(coords)
        
        # Should be close to origin
        assert np.allclose(world_coords[0], [0, 0], atol=0.5)
    
    def test_round_trip_conversion(self, transform):
        """Test that world -> BEV -> world is consistent."""
        # Create test points
        points = np.array([
            [10, 20, 0],
            [-15, 30, 0],
            [5, -10, 0]
        ], dtype=np.float32)
        
        # Convert to BEV and back
        bev_coords = transform.world_to_bev(points)
        world_coords = transform.bev_to_world(bev_coords)
        
        # Should be close to original (within resolution)
        assert np.allclose(points[:, :2], world_coords, 
                          atol=transform.bev_config.resolution)
    
    def test_projection_round_trip(self, transform):
        """Test 3D to 2D projection and back-projection."""
        # Create test points in ego frame
        points_3d = np.array([
            [10, 0, 0],
            [20, 5, 1],
            [15, -3, 0.5]
        ], dtype=np.float32)
        
        # Create synthetic camera parameters
        intrinsics = np.array([
            [1000, 0, 800],
            [0, 1000, 600],
            [0, 0, 1]
        ], dtype=np.float32)
        
        extrinsics = np.eye(4, dtype=np.float32)
        extrinsics[:3, 3] = [0, 0, 1.5]  # Camera 1.5m above ground
        
        # Project to 2D
        points_2d, depth = transform.project_3d_to_2d(
            points_3d, intrinsics, extrinsics
        )
        
        # Back-project to 3D
        points_3d_reconstructed = transform.backproject_2d_to_3d(
            points_2d, depth, intrinsics, extrinsics
        )
        
        # Should match original points
        assert np.allclose(points_3d, points_3d_reconstructed, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
