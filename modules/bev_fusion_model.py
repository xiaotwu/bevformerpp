"""
Complete BEV Fusion Model integrating all components.
Combines LiDAR encoder, camera encoder, spatial fusion, temporal aggregation, and detection head.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from .lidar_encoder import LiDARBEVEncoder
from .camera_encoder import CameraBEVEncoder
from .fusion import SpatialFusionModule
from .temporal_attention import TemporalAggregationModule
from .mc_convrnn import MCConvRNN
from .head import DetectionHead, DetectionPostProcessor
from .data_structures import BEVGridConfig, Box3D


class BEVFusionModel(nn.Module):
    """
    Complete BEV Fusion Model for 3D object detection.
    
    Architecture:
    1. LiDAR BEV Encoder (PointPillars)
    2. Camera BEV Encoder (BEVFormer)
    3. Spatial Fusion (Cross-Attention)
    4. Temporal Aggregation (Transformer-based or MC-ConvRNN)
    5. Detection Head (Classification + Regression)
    
    Implements Requirements 1.1-6.5 from the design document.
    """
    
    def __init__(
        self,
        # BEV grid configuration
        bev_config: Optional[BEVGridConfig] = None,
        
        # Feature dimensions
        lidar_channels: int = 64,
        camera_channels: int = 256,
        fused_channels: int = 256,
        
        # Temporal configuration
        use_temporal_attention: bool = True,
        use_mc_convrnn: bool = False,
        temporal_length: int = 5,
        temporal_hidden_channels: int = 128,
        
        # Detection configuration
        num_classes: int = 10,
        class_names: Optional[List[str]] = None,
        
        # Post-processing
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 100
    ):
        """
        Initialize BEV Fusion Model.
        
        Args:
            bev_config: BEV grid configuration
            lidar_channels: Output channels for LiDAR encoder
            camera_channels: Output channels for camera encoder
            fused_channels: Output channels after fusion
            use_temporal_attention: Whether to use transformer-based temporal aggregation
            use_mc_convrnn: Whether to use MC-ConvRNN temporal aggregation
            temporal_length: Maximum temporal sequence length
            temporal_hidden_channels: Hidden channels for temporal modules
            num_classes: Number of object categories
            class_names: List of class names
            score_threshold: Confidence threshold for detections
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep
        """
        super().__init__()
        
        # Configuration
        self.bev_config = bev_config if bev_config is not None else BEVGridConfig()
        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels
        self.use_temporal_attention = use_temporal_attention
        self.use_mc_convrnn = use_mc_convrnn
        self.num_classes = num_classes
        
        # Get BEV grid size
        self.bev_h, self.bev_w = self.bev_config.grid_size
        
        # 1. LiDAR BEV Encoder
        self.lidar_encoder = LiDARBEVEncoder(
            bev_config=self.bev_config,
            out_channels=lidar_channels
        )
        
        # 2. Camera BEV Encoder
        self.camera_encoder = CameraBEVEncoder(
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            embed_dim=camera_channels
        )
        
        # 3. Spatial Fusion Module
        self.spatial_fusion = SpatialFusionModule(
            lidar_channels=lidar_channels,
            camera_channels=camera_channels,
            fused_channels=fused_channels
        )
        
        # 4. Temporal Aggregation Modules
        if use_temporal_attention:
            self.temporal_attention = TemporalAggregationModule(
                embed_dim=fused_channels,
                num_heads=8,
                max_history=temporal_length,
                use_gating=True
            )
        
        if use_mc_convrnn:
            self.mc_convrnn = MCConvRNN(
                input_channels=fused_channels,
                hidden_channels=temporal_hidden_channels
            )
            self.mc_hidden_state = None  # Will be initialized during forward
        
        # 5. Detection Head
        self.detection_head = DetectionHead(
            in_channels=fused_channels,
            num_classes=num_classes
        )
        
        # 6. Post-processor
        self.post_processor = DetectionPostProcessor(
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            max_detections=max_detections,
            bev_range=(
                self.bev_config.x_min,
                self.bev_config.x_max,
                self.bev_config.y_min,
                self.bev_config.y_max
            ),
            class_names=class_names
        )
    
    def forward(
        self,
        lidar_points: torch.Tensor,
        camera_images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        ego_transform: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BEV Fusion Model.
        
        Args:
            lidar_points: LiDAR point cloud, shape (B, N, 4) or list of (N_i, 4)
            camera_images: Multi-view images, shape (B, N_cam, 3, H, W)
            camera_intrinsics: Camera intrinsics, shape (B, N_cam, 3, 3)
            camera_extrinsics: Camera extrinsics, shape (B, N_cam, 4, 4)
            ego_transform: Ego-motion transform, shape (B, 4, 4)
            return_intermediate: Whether to return intermediate features
        
        Returns:
            Dictionary containing:
            - 'cls_scores': Classification scores
            - 'bbox_preds': Bounding box predictions
            - (optional) intermediate features if return_intermediate=True
        """
        intermediate = {}
        
        # 1. LiDAR BEV Encoding
        lidar_bev = self.lidar_encoder(lidar_points)
        if return_intermediate:
            intermediate['lidar_bev'] = lidar_bev
        
        # 2. Camera BEV Encoding
        camera_bev = self.camera_encoder(
            camera_images,
            camera_intrinsics,
            camera_extrinsics
        )
        if return_intermediate:
            intermediate['camera_bev'] = camera_bev
        
        # 3. Spatial Fusion
        fused_bev = self.spatial_fusion(lidar_bev, camera_bev)
        if return_intermediate:
            intermediate['fused_bev'] = fused_bev
        
        # 4. Temporal Aggregation
        temporal_bev = fused_bev
        
        if self.use_temporal_attention:
            temporal_bev = self.temporal_attention(
                current_bev=temporal_bev,
                ego_transform=ego_transform
            )
            if return_intermediate:
                intermediate['temporal_attention_bev'] = temporal_bev
        
        if self.use_mc_convrnn:
            # Get previous features and hidden state
            # For first frame, these will be None
            prev_features = getattr(self, '_prev_features', None)
            prev_hidden = self.mc_hidden_state
            
            # Forward through MC-ConvRNN
            temporal_bev, new_hidden = self.mc_convrnn(
                current_features=temporal_bev,
                prev_features=prev_features,
                prev_hidden=prev_hidden,
                ego_motion=ego_transform
            )
            
            # Store for next iteration
            self._prev_features = temporal_bev.detach()
            self.mc_hidden_state = new_hidden.detach()
            
            if return_intermediate:
                intermediate['mc_convrnn_bev'] = temporal_bev
        
        # 5. Detection Head
        detections = self.detection_head(temporal_bev)
        
        if return_intermediate:
            return {**detections, 'intermediate': intermediate}
        
        return detections
    
    def predict(
        self,
        lidar_points: torch.Tensor,
        camera_images: torch.Tensor,
        camera_intrinsics: torch.Tensor,
        camera_extrinsics: torch.Tensor,
        ego_transform: Optional[torch.Tensor] = None
    ) -> List[Box3D]:
        """
        Run inference and return decoded detections.
        
        Args:
            lidar_points: LiDAR point cloud
            camera_images: Multi-view images
            camera_intrinsics: Camera intrinsics
            camera_extrinsics: Camera extrinsics
            ego_transform: Ego-motion transform
        
        Returns:
            List of Box3D detections after NMS
        """
        self.eval()
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(
                lidar_points,
                camera_images,
                camera_intrinsics,
                camera_extrinsics,
                ego_transform
            )
            
            # Post-process
            detections = self.post_processor(
                outputs['cls_scores'],
                outputs['bbox_preds']
            )
        
        return detections
    
    def reset_temporal_state(self):
        """Reset temporal state (e.g., at scene boundaries)."""
        if self.use_temporal_attention:
            self.temporal_attention.reset()
        
        if self.use_mc_convrnn:
            self.mc_hidden_state = None
            if hasattr(self, '_prev_features'):
                delattr(self, '_prev_features')
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


def create_bev_fusion_model(config: Dict) -> BEVFusionModel:
    """
    Factory function to create BEV Fusion Model from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Initialized BEVFusionModel
    """
    # Extract BEV grid config
    bev_config = BEVGridConfig(
        x_min=config.get('bev_x_min', -51.2),
        x_max=config.get('bev_x_max', 51.2),
        y_min=config.get('bev_y_min', -51.2),
        y_max=config.get('bev_y_max', 51.2),
        z_min=config.get('bev_z_min', -5.0),
        z_max=config.get('bev_z_max', 3.0),
        resolution=config.get('bev_resolution', 0.2)
    )
    
    # Create model
    model = BEVFusionModel(
        bev_config=bev_config,
        lidar_channels=config.get('lidar_channels', 64),
        camera_channels=config.get('camera_channels', 256),
        fused_channels=config.get('fused_channels', 256),
        use_temporal_attention=config.get('use_temporal_attention', True),
        use_mc_convrnn=config.get('use_mc_convrnn', False),
        temporal_length=config.get('temporal_length', 5),
        temporal_hidden_channels=config.get('temporal_hidden_channels', 128),
        num_classes=config.get('num_classes', 10),
        class_names=config.get('class_names', None),
        score_threshold=config.get('score_threshold', 0.3),
        nms_threshold=config.get('nms_threshold', 0.5),
        max_detections=config.get('max_detections', 100)
    )
    
    return model
