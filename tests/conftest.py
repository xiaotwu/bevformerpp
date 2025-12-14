"""Pytest configuration and fixtures for BEV Fusion System tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
from typing import Tuple

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


@pytest.fixture
def device():
    """Fixture providing the device to use for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Fixture providing default batch size for tests."""
    return 2


@pytest.fixture
def bev_grid_size():
    """Fixture providing BEV grid dimensions."""
    return (200, 200)  # H, W


@pytest.fixture
def sample_point_cloud():
    """Fixture providing a sample point cloud for testing.
    
    Returns:
        np.ndarray: Point cloud of shape (N, 4) with (x, y, z, reflectance)
    """
    n_points = 1000
    x = np.random.uniform(-50, 50, n_points)
    y = np.random.uniform(-50, 50, n_points)
    z = np.random.uniform(-5, 3, n_points)
    r = np.random.uniform(0, 1, n_points)
    return np.stack([x, y, z, r], axis=1).astype(np.float32)


@pytest.fixture
def sample_images(batch_size):
    """Fixture providing sample multi-view images.
    
    Returns:
        torch.Tensor: Images of shape (B, N_cam, 3, H, W)
    """
    n_cameras = 6
    h, w = 900, 1600
    return torch.randn(batch_size, n_cameras, 3, h, w)


@pytest.fixture
def sample_bev_features(batch_size, bev_grid_size):
    """Fixture providing sample BEV features.
    
    Returns:
        torch.Tensor: BEV features of shape (B, C, H, W)
    """
    channels = 64
    h, w = bev_grid_size
    return torch.randn(batch_size, channels, h, w)


@pytest.fixture
def sample_ego_motion(batch_size):
    """Fixture providing sample ego-motion transforms.
    
    Returns:
        torch.Tensor: SE(3) transforms of shape (B, 4, 4)
    """
    transforms = []
    for _ in range(batch_size):
        # Create random SE(3) transform
        transform = torch.eye(4)
        
        # Random translation (small movements)
        transform[:3, 3] = torch.randn(3) * 2.0
        
        # Random rotation around z-axis (yaw)
        yaw = torch.rand(1) * 2 * np.pi - np.pi
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        
        transform[0, 0] = cos_yaw
        transform[0, 1] = -sin_yaw
        transform[1, 0] = sin_yaw
        transform[1, 1] = cos_yaw
        
        transforms.append(transform)
    
    return torch.stack(transforms)


@pytest.fixture
def config_path():
    """Fixture providing path to test configuration."""
    return Path(__file__).parent.parent / "configs" / "base_config.yaml"


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Fixture providing temporary directory for checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir
