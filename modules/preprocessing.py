"""
Data preprocessing utilities for BEV Fusion System.
Includes point cloud filtering, image normalization, and BEV coordinate transformations.
"""

import numpy as np
import torch
from typing import Tuple, Optional
import cv2

from .data_structures import BEVGridConfig


class PointCloudPreprocessor:
    """Preprocessing utilities for LiDAR point clouds."""
    
    def __init__(self, bev_config: BEVGridConfig):
        """
        Initialize point cloud preprocessor.
        
        Args:
            bev_config: BEV grid configuration
        """
        self.bev_config = bev_config
    
    def filter_points(self, points: np.ndarray) -> np.ndarray:
        """
        Filter point cloud to keep only points within BEV range.
        
        Args:
            points: Point cloud array of shape (N, 4) [x, y, z, intensity]
            
        Returns:
            Filtered point cloud array
        """
        # Filter by x range
        mask_x = (points[:, 0] >= self.bev_config.x_min) & \
                 (points[:, 0] <= self.bev_config.x_max)
        
        # Filter by y range
        mask_y = (points[:, 1] >= self.bev_config.y_min) & \
                 (points[:, 1] <= self.bev_config.y_max)
        
        # Filter by z range
        mask_z = (points[:, 2] >= self.bev_config.z_min) & \
                 (points[:, 2] <= self.bev_config.z_max)
        
        # Combine masks
        mask = mask_x & mask_y & mask_z
        
        return points[mask]
    
    def normalize_points(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize point cloud coordinates and intensity.
        
        Args:
            points: Point cloud array of shape (N, 4) [x, y, z, intensity]
            
        Returns:
            Normalized point cloud array
        """
        normalized = points.copy()
        
        # Normalize spatial coordinates to [-1, 1]
        normalized[:, 0] = (points[:, 0] - self.bev_config.x_min) / \
                          (self.bev_config.x_max - self.bev_config.x_min) * 2 - 1
        normalized[:, 1] = (points[:, 1] - self.bev_config.y_min) / \
                          (self.bev_config.y_max - self.bev_config.y_min) * 2 - 1
        normalized[:, 2] = (points[:, 2] - self.bev_config.z_min) / \
                          (self.bev_config.z_max - self.bev_config.z_min) * 2 - 1
        
        # Normalize intensity to [0, 1] (assuming intensity is already in reasonable range)
        # nuScenes intensity is typically in [0, 255]
        normalized[:, 3] = np.clip(points[:, 3] / 255.0, 0, 1)
        
        return normalized
    
    def points_to_bev_coords(self, points: np.ndarray) -> np.ndarray:
        """
        Convert point cloud coordinates to BEV grid coordinates.
        
        Args:
            points: Point cloud array of shape (N, 4) [x, y, z, intensity]
            
        Returns:
            BEV coordinates array of shape (N, 2) [grid_x, grid_y]
        """
        # Convert to grid indices
        grid_x = ((points[:, 0] - self.bev_config.x_min) / 
                  self.bev_config.resolution).astype(np.int32)
        grid_y = ((points[:, 1] - self.bev_config.y_min) / 
                  self.bev_config.resolution).astype(np.int32)
        
        # Clip to valid range
        H, W = self.bev_config.grid_size
        grid_x = np.clip(grid_x, 0, W - 1)
        grid_y = np.clip(grid_y, 0, H - 1)
        
        return np.stack([grid_x, grid_y], axis=1)
    
    def preprocess(self, points: np.ndarray, 
                   filter: bool = True,
                   normalize: bool = False) -> np.ndarray:
        """
        Apply full preprocessing pipeline to point cloud.
        
        Args:
            points: Point cloud array of shape (N, 4)
            filter: Whether to filter points by BEV range
            normalize: Whether to normalize coordinates
            
        Returns:
            Preprocessed point cloud array
        """
        if filter:
            points = self.filter_points(points)
        
        if normalize:
            points = self.normalize_points(points)
        
        return points


class ImagePreprocessor:
    """Preprocessing utilities for camera images."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (256, 704),
                 mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                 std: Tuple[float, float, float] = (0.229, 0.224, 0.225)):
        """
        Initialize image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            mean: Mean values for normalization (ImageNet default)
            std: Standard deviation values for normalization (ImageNet default)
        """
        self.target_size = target_size
        self.mean = np.array(mean).reshape(1, 1, 3)
        self.std = np.array(std).reshape(1, 1, 3)
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Image array of shape (H, W, 3)
            
        Returns:
            Resized image array
        """
        # OpenCV uses (width, height) order
        target_width, target_height = self.target_size[1], self.target_size[0]
        resized = cv2.resize(image, (target_width, target_height), 
                            interpolation=cv2.INTER_LINEAR)
        return resized
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image using ImageNet statistics.
        
        Args:
            image: Image array of shape (H, W, 3) with values in [0, 255]
            
        Returns:
            Normalized image array with values approximately in [-2, 2]
        """
        # Convert to float and scale to [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Normalize using mean and std (ensure float32)
        mean_f32 = self.mean.astype(np.float32)
        std_f32 = self.std.astype(np.float32)
        normalized = ((image_float - mean_f32) / std_f32).astype(np.float32)
        
        return normalized
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to PyTorch tensor with channels first.
        
        Args:
            image: Image array of shape (H, W, 3)
            
        Returns:
            Tensor of shape (3, H, W)
        """
        # Transpose to (C, H, W)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return tensor
    
    def adjust_intrinsics(self, 
                         intrinsics: np.ndarray,
                         original_size: Tuple[int, int]) -> np.ndarray:
        """
        Adjust camera intrinsics for image resizing.
        
        Args:
            intrinsics: Original intrinsic matrix (3, 3)
            original_size: Original image size (height, width)
            
        Returns:
            Adjusted intrinsic matrix
        """
        adjusted = intrinsics.copy()
        
        # Calculate scale factors
        scale_x = self.target_size[1] / original_size[1]  # width scale
        scale_y = self.target_size[0] / original_size[0]  # height scale
        
        # Adjust focal lengths and principal point
        adjusted[0, 0] *= scale_x  # fx
        adjusted[0, 2] *= scale_x  # cx
        adjusted[1, 1] *= scale_y  # fy
        adjusted[1, 2] *= scale_y  # cy
        
        return adjusted
    
    def preprocess(self, 
                   image: np.ndarray,
                   resize: bool = True,
                   normalize: bool = True,
                   to_tensor: bool = True) -> np.ndarray:
        """
        Apply full preprocessing pipeline to image.
        
        Args:
            image: Image array of shape (H, W, 3)
            resize: Whether to resize image
            normalize: Whether to normalize image
            to_tensor: Whether to convert to tensor
            
        Returns:
            Preprocessed image (tensor if to_tensor=True, else array)
        """
        if resize:
            image = self.resize(image)
        
        if normalize:
            image = self.normalize(image)
        
        if to_tensor:
            image = self.to_tensor(image)
        
        return image


class BEVTransform:
    """Coordinate transformation utilities for BEV space."""
    
    def __init__(self, bev_config: BEVGridConfig):
        """
        Initialize BEV transform utilities.
        
        Args:
            bev_config: BEV grid configuration
        """
        self.bev_config = bev_config
    
    def world_to_bev(self, points: np.ndarray) -> np.ndarray:
        """
        Transform world coordinates to BEV grid coordinates.
        
        Args:
            points: Points in world frame (N, 2) or (N, 3) [x, y] or [x, y, z]
            
        Returns:
            BEV grid coordinates (N, 2) [u, v] where u, v are pixel indices
        """
        # Extract x, y coordinates
        if points.shape[1] == 3:
            xy = points[:, :2]
        else:
            xy = points
        
        # Convert to grid coordinates
        u = ((xy[:, 0] - self.bev_config.x_min) / 
             self.bev_config.resolution).astype(np.int32)
        v = ((xy[:, 1] - self.bev_config.y_min) / 
             self.bev_config.resolution).astype(np.int32)
        
        return np.stack([u, v], axis=1)
    
    def bev_to_world(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform BEV grid coordinates to world coordinates.
        
        Args:
            coords: BEV grid coordinates (N, 2) [u, v]
            
        Returns:
            World coordinates (N, 2) [x, y]
        """
        x = coords[:, 0] * self.bev_config.resolution + self.bev_config.x_min
        y = coords[:, 1] * self.bev_config.resolution + self.bev_config.y_min
        
        return np.stack([x, y], axis=1)
    
    def create_bev_grid(self) -> np.ndarray:
        """
        Create a meshgrid of BEV coordinates in world frame.
        
        Returns:
            Grid of shape (H, W, 2) containing [x, y] coordinates for each pixel
        """
        H, W = self.bev_config.grid_size
        
        # Create coordinate arrays
        x = np.linspace(self.bev_config.x_min, self.bev_config.x_max, W)
        y = np.linspace(self.bev_config.y_min, self.bev_config.y_max, H)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x, y)
        
        # Stack to (H, W, 2)
        grid = np.stack([xx, yy], axis=2)
        
        return grid
    
    def apply_ego_motion(self, 
                        points: np.ndarray,
                        ego_transform: np.ndarray) -> np.ndarray:
        """
        Apply ego-motion transformation to points.
        
        Args:
            points: Points in frame t (N, 3) [x, y, z]
            ego_transform: SE(3) transform from frame t to frame t+1 (4, 4)
            
        Returns:
            Transformed points in frame t+1 (N, 3)
        """
        # Convert to homogeneous coordinates
        N = points.shape[0]
        points_homo = np.concatenate([points, np.ones((N, 1))], axis=1)
        
        # Apply transformation
        points_transformed = (ego_transform @ points_homo.T).T
        
        # Convert back to 3D
        return points_transformed[:, :3]
    
    def project_3d_to_2d(self,
                        points_3d: np.ndarray,
                        intrinsics: np.ndarray,
                        extrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points in ego frame (N, 3) [x, y, z]
            intrinsics: Camera intrinsic matrix (3, 3)
            extrinsics: Camera extrinsic matrix (4, 4) - ego to camera
            
        Returns:
            Tuple of:
                - 2D points (N, 2) [u, v]
                - Depth values (N,)
        """
        # Transform to camera frame
        N = points_3d.shape[0]
        points_homo = np.concatenate([points_3d, np.ones((N, 1))], axis=1)
        
        # ego to camera transform
        ego_to_cam = np.linalg.inv(extrinsics)
        points_cam = (ego_to_cam @ points_homo.T).T[:, :3]
        
        # Project to image plane
        points_2d_homo = (intrinsics @ points_cam.T).T
        
        # Normalize by depth
        depth = points_2d_homo[:, 2]
        points_2d = points_2d_homo[:, :2] / depth[:, np.newaxis]
        
        return points_2d, depth
    
    def backproject_2d_to_3d(self,
                            points_2d: np.ndarray,
                            depth: np.ndarray,
                            intrinsics: np.ndarray,
                            extrinsics: np.ndarray) -> np.ndarray:
        """
        Back-project 2D image points to 3D ego frame.
        
        Args:
            points_2d: 2D points (N, 2) [u, v]
            depth: Depth values (N,)
            intrinsics: Camera intrinsic matrix (3, 3)
            extrinsics: Camera extrinsic matrix (4, 4) - ego to camera
            
        Returns:
            3D points in ego frame (N, 3) [x, y, z]
        """
        # Convert to homogeneous coordinates
        N = points_2d.shape[0]
        points_2d_homo = np.concatenate([points_2d, np.ones((N, 1))], axis=1)
        
        # Scale by depth
        points_2d_homo = points_2d_homo * depth[:, np.newaxis]
        
        # Back-project to camera frame
        intrinsics_inv = np.linalg.inv(intrinsics)
        points_cam = (intrinsics_inv @ points_2d_homo.T).T
        
        # Transform to ego frame
        points_cam_homo = np.concatenate([points_cam, np.ones((N, 1))], axis=1)
        points_ego = (extrinsics @ points_cam_homo.T).T[:, :3]
        
        return points_ego
