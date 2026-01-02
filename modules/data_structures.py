"""
Data structures for BEV Fusion System.
Defines Sample and Box3D dataclasses as specified in the design document.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import numpy as np


@dataclass
class Box3D:
    """
    3D bounding box representation.
    
    Attributes:
        center: (3,) array [x, y, z] in meters
        size: (3,) array [width, length, height] in meters
        yaw: rotation around z-axis in radians
        label: object category (e.g., 'car', 'pedestrian')
        score: confidence score (default 1.0 for ground truth)
        token: annotation token from dataset
    """
    center: np.ndarray  # (3,) [x, y, z]
    size: np.ndarray    # (3,) [width, length, height]
    yaw: float          # rotation around z-axis
    label: str          # object category
    score: float = 1.0  # confidence score
    token: str = ""     # annotation token
    
    def __post_init__(self):
        """Validate box parameters."""
        assert self.center.shape == (3,), f"Center must be (3,), got {self.center.shape}"
        assert self.size.shape == (3,), f"Size must be (3,), got {self.size.shape}"
        assert np.all(self.size > 0), "Size must be positive"
        assert 0 <= self.score <= 1, f"Score must be in [0, 1], got {self.score}"


@dataclass
class Sample:
    """
    A single sample from the nuScenes dataset.
    
    Attributes:
        sample_token: Unique identifier for this sample
        scene_token: Scene this sample belongs to
        timestamp: Timestamp in microseconds
        lidar_path: Path to LiDAR point cloud file
        lidar_points: Optional loaded point cloud (N, 4) [x, y, z, intensity]
        camera_paths: Dictionary mapping camera name to image path
        camera_images: Optional loaded images
        camera_intrinsics: Dictionary mapping camera name to (3, 3) intrinsic matrix
        camera_extrinsics: Dictionary mapping camera name to (4, 4) extrinsic matrix (sensor to ego)
        ego_pose: (4, 4) SE(3) transform from ego to global frame
        annotations: List of Box3D objects (ground truth)
    """
    # Identifiers
    sample_token: str
    scene_token: str
    timestamp: int
    
    # LiDAR data
    lidar_path: str
    lidar_points: Optional[np.ndarray] = None  # (N, 4)
    
    # Camera data
    camera_paths: Dict[str, str] = field(default_factory=dict)  # 6 cameras
    camera_images: Optional[Dict[str, np.ndarray]] = None
    camera_intrinsics: Dict[str, np.ndarray] = field(default_factory=dict)  # (3, 3)
    camera_extrinsics: Dict[str, np.ndarray] = field(default_factory=dict)  # (4, 4)
    
    # Ego-motion
    ego_pose: Optional[np.ndarray] = None  # (4, 4) SE(3) transform
    
    # Annotations
    annotations: List[Box3D] = field(default_factory=list)


@dataclass
class BEVGridConfig:
    """
    Configuration for BEV grid.

    Attributes:
        x_min, x_max: BEV range in x direction (meters)
        y_min, y_max: BEV range in y direction (meters)
        z_min, z_max: BEV range in z direction (meters)
        resolution: meters per pixel

    IMPORTANT: The default resolution (0.2) produces a 512×512 grid.
    For training with 200×200 targets, use resolution=0.512:

        config = BEVGridConfig(resolution=0.512)  # Produces 200×200 grid

    Alternatively, use the class method for explicit grid size:

        config = BEVGridConfig.from_grid_size(bev_h=200, bev_w=200)
    """
    x_min: float = -51.2  # meters
    x_max: float = 51.2
    y_min: float = -51.2
    y_max: float = 51.2
    z_min: float = -5.0
    z_max: float = 3.0
    resolution: float = 0.2  # meters per pixel

    def __post_init__(self):
        """Validate configuration."""
        assert self.x_max > self.x_min, "x_max must be greater than x_min"
        assert self.y_max > self.y_min, "y_max must be greater than y_min"
        assert self.z_max > self.z_min, "z_max must be greater than z_min"
        assert self.resolution > 0, "resolution must be positive"

    @classmethod
    def from_grid_size(cls, bev_h: int = 200, bev_w: int = 200,
                       x_range: tuple = (-51.2, 51.2),
                       y_range: tuple = (-51.2, 51.2),
                       z_range: tuple = (-5.0, 3.0)) -> 'BEVGridConfig':
        """
        Create config from explicit grid size.

        Args:
            bev_h: BEV grid height (pixels)
            bev_w: BEV grid width (pixels)
            x_range: (x_min, x_max) in meters
            y_range: (y_min, y_max) in meters
            z_range: (z_min, z_max) in meters

        Returns:
            BEVGridConfig with computed resolution
        """
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        # Compute resolution to match requested grid size
        resolution_x = (x_max - x_min) / bev_w
        resolution_y = (y_max - y_min) / bev_h

        # Use the larger resolution (coarser) to ensure grid fits
        resolution = max(resolution_x, resolution_y)

        return cls(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_min=z_min, z_max=z_max,
            resolution=resolution
        )
    
    @property
    def grid_size(self) -> tuple:
        """Calculate grid size (H, W) from range and resolution."""
        H = int((self.y_max - self.y_min) / self.resolution)
        W = int((self.x_max - self.x_min) / self.resolution)
        return (H, W)
    
    @property
    def x_range(self) -> tuple:
        """Get x range as tuple."""
        return (self.x_min, self.x_max)
    
    @property
    def y_range(self) -> tuple:
        """Get y range as tuple."""
        return (self.y_min, self.y_max)
    
    @property
    def z_range(self) -> tuple:
        """Get z range as tuple."""
        return (self.z_min, self.z_max)

    @property
    def bev_range(self) -> tuple:
        """
        Get BEV range as (x_min, x_max, y_min, y_max) tuple.

        This format is used by warping utilities like align_bev_features,
        generate_grid_from_transform, and compute_visibility_mask.

        Example:
            >>> config = BEVGridConfig.from_grid_size(bev_h=200, bev_w=200)
            >>> temporal_module = TemporalAggregationModule(bev_range=config.bev_range)
        """
        return (self.x_min, self.x_max, self.y_min, self.y_max)

    @property
    def z_ref(self) -> float:
        """
        Reference height for BEV plane projection (midpoint of z range).

        This is the height at which BEV grid points are sampled for camera projection.
        Using the midpoint ensures alignment between camera and LiDAR BEV features.

        Returns:
            z_ref in meters
        """
        return (self.z_min + self.z_max) / 2

    def to_dict(self) -> dict:
        """
        Convert to dictionary format for camera encoder bev_config.

        Returns dict with keys: 'x_min', 'x_max', 'y_min', 'y_max', 'z_ref'
        """
        return {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'z_ref': self.z_ref  # Use property for consistency
        }
