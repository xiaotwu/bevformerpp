"""
nuScenes dataset loader for BEV Fusion System.
Implements loading of LiDAR, multi-view images, calibration, and annotations.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pyquaternion import Quaternion

from .data_structures import Sample, Box3D, BEVGridConfig


class NuScenesLoader:
    """
    Loader for nuScenes dataset metadata.
    Provides methods to load and query dataset information.
    """
    
    def __init__(self, dataroot: str = "data", version: str = "v1.0-mini"):
        """
        Initialize nuScenes loader.
        
        Args:
            dataroot: Root directory containing nuScenes data
            version: Dataset version (e.g., 'v1.0-mini', 'v1.0-trainval')
        """
        self.dataroot = Path(dataroot)
        self.version = version
        self.version_path = self.dataroot / version
        
        # Load all metadata tables
        self.tables = {}
        self._load_tables()
        
        # Create token-to-index mappings for fast lookup
        self._create_indices()
    
    def _load_tables(self):
        """Load all JSON metadata tables."""
        table_names = [
            'attribute', 'calibrated_sensor', 'category', 'ego_pose',
            'instance', 'log', 'map', 'sample', 'sample_annotation',
            'sample_data', 'scene', 'sensor', 'visibility'
        ]
        
        for table_name in table_names:
            table_path = self.version_path / f"{table_name}.json"
            with open(table_path, 'r') as f:
                self.tables[table_name] = json.load(f)
    
    def _create_indices(self):
        """Create token-to-index mappings for fast lookup."""
        self.indices = {}
        for table_name, records in self.tables.items():
            self.indices[table_name] = {
                record['token']: idx for idx, record in enumerate(records)
            }
    
    def get(self, table_name: str, token: str) -> Dict:
        """
        Get a record by token.
        
        Args:
            table_name: Name of the table
            token: Token to look up
            
        Returns:
            Record dictionary
        """
        idx = self.indices[table_name][token]
        return self.tables[table_name][idx]
    
    def get_sample_data_path(self, sample_data_token: str) -> str:
        """Get the full path to a sample data file."""
        sd = self.get('sample_data', sample_data_token)
        return str(self.dataroot / sd['filename'])


class NuScenesDataset(Dataset):
    """
    PyTorch Dataset for nuScenes.
    Loads synchronized multi-sensor data with temporal sequences.
    """
    
    # Camera names in nuScenes
    CAMERA_NAMES = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
        'CAM_BACK_LEFT'
    ]
    
    def __init__(
        self,
        dataroot: str = "data",
        version: str = "v1.0-mini",
        split: str = "train",
        sequence_length: int = 1,
        load_lidar: bool = True,
        load_images: bool = True,
        load_annotations: bool = True,
        bev_config: Optional[BEVGridConfig] = None
    ):
        """
        Initialize nuScenes dataset.
        
        Args:
            dataroot: Root directory containing nuScenes data
            version: Dataset version
            split: Data split ('train', 'val', 'test')
            sequence_length: Number of frames in temporal sequence
            load_lidar: Whether to load LiDAR point clouds
            load_images: Whether to load camera images
            load_annotations: Whether to load ground truth annotations
            bev_config: BEV grid configuration
        """
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.sequence_length = sequence_length
        self.load_lidar = load_lidar
        self.load_images = load_images
        self.load_annotations = load_annotations
        self.bev_config = bev_config or BEVGridConfig()
        
        # Initialize loader
        self.loader = NuScenesLoader(dataroot, version)
        
        # Get list of samples
        self.samples = self._get_samples()
        
        print(f"Loaded {len(self.samples)} samples from nuScenes {version} ({split})")
    
    def _get_samples(self) -> List[str]:
        """
        Get list of sample tokens for this split.
        For v1.0-mini, we use a simple split based on scene.
        """
        # Get all samples
        all_samples = self.loader.tables['sample']
        
        # For v1.0-mini, split by scene
        # Scenes 0-7 for train, 8-9 for val
        scenes = self.loader.tables['scene']
        
        if self.split == 'train':
            scene_tokens = [s['token'] for s in scenes[:8]]
        elif self.split == 'val':
            scene_tokens = [s['token'] for s in scenes[8:]]
        else:  # test or all
            scene_tokens = [s['token'] for s in scenes]
        
        # Filter samples by scene
        sample_tokens = [
            s['token'] for s in all_samples
            if s['scene_token'] in scene_tokens
        ]
        
        return sample_tokens
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample with temporal sequence.
        
        Returns:
            Dictionary containing:
                - samples: List of Sample objects (length = sequence_length)
                - ego_transforms: List of ego-motion transforms between frames
        """
        # Get current sample token
        current_token = self.samples[idx]
        
        # Build temporal sequence
        sequence_tokens = self._build_sequence(current_token)
        
        # Load samples
        samples = [self._load_sample(token) for token in sequence_tokens]
        
        # Compute ego-motion transforms
        ego_transforms = self._compute_ego_transforms(samples)
        
        return {
            'samples': samples,
            'ego_transforms': ego_transforms
        }
    
    def _build_sequence(self, current_token: str) -> List[str]:
        """
        Build temporal sequence ending at current token.
        
        Args:
            current_token: Token of the current (last) sample
            
        Returns:
            List of sample tokens (oldest to newest)
        """
        tokens = []
        token = current_token
        
        # Traverse backwards
        for _ in range(self.sequence_length):
            tokens.insert(0, token)
            sample = self.loader.get('sample', token)
            
            # Move to previous sample
            if sample['prev'] != '':
                token = sample['prev']
            else:
                # No more previous samples, pad with current
                break
        
        # Pad if sequence is shorter than requested
        while len(tokens) < self.sequence_length:
            tokens.insert(0, tokens[0])
        
        return tokens
    
    def _get_sample_data_tokens(self, sample_token: str) -> Dict[str, str]:
        """
        Get sample_data tokens for all sensors in a sample.
        
        Args:
            sample_token: Token of the sample
            
        Returns:
            Dictionary mapping sensor name to sample_data token
        """
        # Find all sample_data records that belong to this sample
        sensor_tokens = {}
        for sd in self.loader.tables['sample_data']:
            if sd['sample_token'] == sample_token and sd['is_key_frame']:
                # Get sensor name from calibrated_sensor
                cs = self.loader.get('calibrated_sensor', sd['calibrated_sensor_token'])
                sensor = self.loader.get('sensor', cs['sensor_token'])
                sensor_name = sensor['channel']
                sensor_tokens[sensor_name] = sd['token']
        
        return sensor_tokens
    
    def _load_sample(self, sample_token: str) -> Sample:
        """
        Load a single sample with all sensor data.
        
        Args:
            sample_token: Token of the sample to load
            
        Returns:
            Sample object with loaded data
        """
        sample_record = self.loader.get('sample', sample_token)
        
        # Get sensor data tokens
        sensor_tokens = self._get_sample_data_tokens(sample_token)
        
        # Get LiDAR path first (required field)
        lidar_token = sensor_tokens['LIDAR_TOP']
        lidar_path = self.loader.get_sample_data_path(lidar_token)
        
        # Create Sample object
        sample = Sample(
            sample_token=sample_token,
            scene_token=sample_record['scene_token'],
            timestamp=sample_record['timestamp'],
            lidar_path=lidar_path
        )
        
        # Load LiDAR data
        if self.load_lidar:
            sample.lidar_points = self._load_lidar_points(sample.lidar_path)
        
        # Load camera data
        if self.load_images:
            for cam_name in self.CAMERA_NAMES:
                cam_token = sensor_tokens[cam_name]
                cam_sd = self.loader.get('sample_data', cam_token)
                
                # Image path
                img_path = self.loader.get_sample_data_path(cam_token)
                sample.camera_paths[cam_name] = img_path
                
                # Load image (optional, can be lazy loaded)
                # For now, we'll load it
                if sample.camera_images is None:
                    sample.camera_images = {}
                sample.camera_images[cam_name] = self._load_image(img_path)
                
                # Calibration
                cs_token = cam_sd['calibrated_sensor_token']
                cs = self.loader.get('calibrated_sensor', cs_token)
                
                # Intrinsics
                sample.camera_intrinsics[cam_name] = np.array(cs['camera_intrinsic'])
                
                # Extrinsics (sensor to ego)
                rotation = Quaternion(cs['rotation']).rotation_matrix
                translation = np.array(cs['translation'])
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = rotation
                extrinsic[:3, 3] = translation
                sample.camera_extrinsics[cam_name] = extrinsic
        
        # Load ego pose (use LIDAR_TOP timestamp as reference)
        lidar_sd = self.loader.get('sample_data', lidar_token)
        ego_pose_token = lidar_sd['ego_pose_token']
        ego_pose_record = self.loader.get('ego_pose', ego_pose_token)
        
        rotation = Quaternion(ego_pose_record['rotation']).rotation_matrix
        translation = np.array(ego_pose_record['translation'])
        ego_pose = np.eye(4)
        ego_pose[:3, :3] = rotation
        ego_pose[:3, 3] = translation
        sample.ego_pose = ego_pose
        
        # Load annotations
        if self.load_annotations:
            sample.annotations = self._load_annotations(sample_record, ego_pose)
        
        return sample
    
    def _load_lidar_points(self, lidar_path: str) -> np.ndarray:
        """
        Load LiDAR point cloud from binary file.
        
        Args:
            lidar_path: Path to .pcd.bin file
            
        Returns:
            Point cloud array of shape (N, 4) [x, y, z, intensity]
        """
        points = np.fromfile(lidar_path, dtype=np.float32)
        points = points.reshape(-1, 5)  # nuScenes has 5 channels
        # Return only x, y, z, intensity (drop ring index)
        return points[:, :4]
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        Load camera image.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Image array of shape (H, W, 3)
        """
        img = Image.open(img_path)
        return np.array(img)
    
    def _load_annotations(self, sample_record: Dict, ego_pose: np.ndarray) -> List[Box3D]:
        """
        Load ground truth annotations for a sample.
        
        Args:
            sample_record: Sample record from metadata
            ego_pose: Ego pose matrix (ego to global)
            
        Returns:
            List of Box3D objects in ego frame
        """
        annotations = []
        
        # Find all annotations for this sample
        sample_token = sample_record['token']
        ann_records = [ann for ann in self.loader.tables['sample_annotation'] 
                      if ann['sample_token'] == sample_token]
        
        for ann in ann_records:
            
            # Get box in global frame
            center_global = np.array(ann['translation'])
            size = np.array(ann['size'])  # [width, length, height]
            rotation = Quaternion(ann['rotation'])
            
            # Convert to ego frame
            # ego_pose is ego -> global, so we need global -> ego = inv(ego_pose)
            ego_to_global = ego_pose
            global_to_ego = np.linalg.inv(ego_to_global)
            
            # Transform center
            center_global_homo = np.append(center_global, 1.0)
            center_ego_homo = global_to_ego @ center_global_homo
            center_ego = center_ego_homo[:3]
            
            # Transform rotation
            # Extract rotation from ego_to_global
            ego_rotation = Quaternion(matrix=ego_to_global[:3, :3])
            # Rotation in ego frame = inv(ego_rotation) * global_rotation
            rotation_ego = ego_rotation.inverse * rotation
            
            # Get yaw angle (rotation around z-axis)
            yaw = rotation_ego.yaw_pitch_roll[0]
            
            # Get category through instance
            instance_token = ann['instance_token']
            instance = self.loader.get('instance', instance_token)
            category_token = instance['category_token']
            category = self.loader.get('category', category_token)
            label = category['name']
            
            # Create Box3D
            box = Box3D(
                center=center_ego,
                size=size,
                yaw=yaw,
                label=label,
                score=1.0,  # Ground truth
                token=ann['token']
            )
            
            annotations.append(box)
        
        return annotations
    
    def _compute_ego_transforms(self, samples: List[Sample]) -> List[np.ndarray]:
        """
        Compute ego-motion transforms between consecutive frames.
        
        Args:
            samples: List of Sample objects
            
        Returns:
            List of SE(3) transforms from frame i to frame i+1
            Length is len(samples) - 1
        """
        transforms = []
        
        for i in range(len(samples) - 1):
            # Transform from frame i to frame i+1
            # ego_pose_i: ego_i -> global
            # ego_pose_{i+1}: ego_{i+1} -> global
            # Transform: ego_i -> global -> ego_{i+1}
            # = inv(ego_pose_{i+1}) @ ego_pose_i
            
            ego_i_to_global = samples[i].ego_pose
            ego_next_to_global = samples[i + 1].ego_pose
            
            global_to_ego_next = np.linalg.inv(ego_next_to_global)
            transform = global_to_ego_next @ ego_i_to_global
            
            transforms.append(transform)
        
        return transforms


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching samples.
    Converts Sample objects to tensors expected by the model.
    
    Args:
        batch: List of sample dictionaries from __getitem__
        
    Returns:
        Batched dictionary with tensors:
            - img: (B, T, N_cams, C, H, W)
            - intrinsics: (B, N_cams, 3, 3)
            - extrinsics: (B, N_cams, 4, 4)
            - ego_pose: (B, 4, 4)
    """
    import torch
    import torch.nn.functional as F
    
    CAMERA_NAMES = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT'
    ]
    IMG_SIZE = (224, 400)  # (H, W) - resize for memory efficiency
    
    batch_imgs = []
    batch_intrinsics = []
    batch_extrinsics = []
    batch_ego_poses = []
    
    for item in batch:
        samples = item['samples']  # List of Sample objects (temporal sequence)
        
        # Stack images across time steps
        seq_imgs = []
        for sample in samples:
            cam_imgs = []
            for cam_name in CAMERA_NAMES:
                if sample.camera_images and cam_name in sample.camera_images:
                    img = sample.camera_images[cam_name]  # (H, W, 3) numpy
                    # Normalize to [0, 1] and convert to (C, H, W)
                    img = torch.from_numpy(img).float() / 255.0
                    img = img.permute(2, 0, 1)  # (3, H, W)
                    # Resize for memory efficiency
                    img = F.interpolate(img.unsqueeze(0), size=IMG_SIZE, mode='bilinear', align_corners=False).squeeze(0)
                else:
                    img = torch.zeros(3, IMG_SIZE[0], IMG_SIZE[1])
                cam_imgs.append(img)
            seq_imgs.append(torch.stack(cam_imgs, dim=0))  # (N_cams, 3, H, W)
        batch_imgs.append(torch.stack(seq_imgs, dim=0))  # (T, N_cams, 3, H, W)
        
        # Use last sample for intrinsics/extrinsics/ego_pose
        last_sample = samples[-1]
        
        # Intrinsics
        intrinsics = []
        for cam_name in CAMERA_NAMES:
            if cam_name in last_sample.camera_intrinsics:
                K = torch.from_numpy(last_sample.camera_intrinsics[cam_name]).float()
            else:
                K = torch.eye(3)
            intrinsics.append(K)
        batch_intrinsics.append(torch.stack(intrinsics, dim=0))  # (N_cams, 3, 3)
        
        # Extrinsics
        extrinsics = []
        for cam_name in CAMERA_NAMES:
            if cam_name in last_sample.camera_extrinsics:
                E = torch.from_numpy(last_sample.camera_extrinsics[cam_name]).float()
            else:
                E = torch.eye(4)
            extrinsics.append(E)
        batch_extrinsics.append(torch.stack(extrinsics, dim=0))  # (N_cams, 4, 4)
        
        # Ego pose
        if last_sample.ego_pose is not None:
            ego = torch.from_numpy(last_sample.ego_pose).float()
        else:
            ego = torch.eye(4)
        batch_ego_poses.append(ego)
    
    return {
        'img': torch.stack(batch_imgs, dim=0),  # (B, T, N_cams, C, H, W)
        'intrinsics': torch.stack(batch_intrinsics, dim=0),  # (B, N_cams, 3, 3)
        'extrinsics': torch.stack(batch_extrinsics, dim=0),  # (B, N_cams, 4, 4)
        'ego_pose': torch.stack(batch_ego_poses, dim=0),  # (B, 4, 4)
    }
