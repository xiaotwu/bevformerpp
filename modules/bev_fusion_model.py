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

        # Spatial fusion configuration
        fusion_type: str = "bidirectional_cross_attn",  # PROPOSAL DEFAULT
        fusion_window_size: int = 7,
        fusion_use_bidirectional: bool = True,
        fusion_use_gate: bool = True,
        fusion_pos_encoding: str = "sinusoidal_2d",
        fusion_dropout: float = 0.0,
        fusion_token_downsample: int = 4,  # FIX 3: Downsample before cross-attention

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
            fusion_type: Type of spatial fusion:
                - "bidirectional_cross_attn": Bidirectional cross-attention (PROPOSAL DEFAULT)
                - "cross_attention": Legacy unidirectional cross-attention
                - "local_attention": Local window attention (recommended for production)
                - "convolutional": Conv-based fusion (fastest, lowest memory)
            fusion_window_size: Window size for local attention (if fusion_type="local_attention")
            fusion_use_bidirectional: Enable both directions in bidirectional_cross_attn
            fusion_use_gate: Enable learned gating in bidirectional_cross_attn
            fusion_pos_encoding: Positional encoding type ("sinusoidal_2d" or "none")
            fusion_dropout: Dropout rate for fusion attention layers
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
            config=self.bev_config,
            out_channels=lidar_channels
        )

        # 2. Camera BEV Encoder
        # P0-2 FIX: Pass unified BEV grid config including z_ref for geometric alignment
        bev_dict = self.bev_config.to_dict()  # Gets x_min, x_max, y_min, y_max, z_ref
        self.camera_encoder = CameraBEVEncoder(
            bev_h=self.bev_h,
            bev_w=self.bev_w,
            bev_z_ref=bev_dict['z_ref'],  # CRITICAL: Use unified z_ref
            embed_dim=camera_channels,
            bev_x_range=(bev_dict['x_min'], bev_dict['x_max']),
            bev_y_range=(bev_dict['y_min'], bev_dict['y_max'])
        )
        
        # 3. Spatial Fusion Module
        self.spatial_fusion = SpatialFusionModule(
            lidar_channels=lidar_channels,
            camera_channels=camera_channels,
            fused_channels=fused_channels,
            fusion_type=fusion_type,
            window_size=fusion_window_size,
            use_bidirectional=fusion_use_bidirectional,
            use_gate=fusion_use_gate,
            pos_encoding=fusion_pos_encoding,
            dropout=fusion_dropout,
            token_downsample=fusion_token_downsample
        )
        
        # 4. Temporal Aggregation Modules
        # Pass bev_range from config to ensure consistent warping
        if use_temporal_attention:
            self.temporal_attention = TemporalAggregationModule(
                embed_dim=fused_channels,
                num_heads=8,
                max_history=temporal_length,
                use_gating=True,
                bev_range=self.bev_config.bev_range
            )

        if use_mc_convrnn:
            self.mc_convrnn = MCConvRNN(
                input_channels=fused_channels,
                hidden_channels=temporal_hidden_channels,
                bev_range=self.bev_config.bev_range
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
        lidar_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BEV Fusion Model.

        Handles both single-frame and temporal inputs automatically:
        - If inputs have temporal dimension (B, T, ...) with T>1, delegates to forward_sequence()
        - If T==1, squeezes temporal dimension and proceeds with single-step logic
        - If inputs are already 3D/4D without temporal dim, processes directly

        Args:
            lidar_points: LiDAR point cloud
                - Shape (B, N, 4): single frame batch
                - Shape (B, T, N, 4): temporal sequence (T may be 1)
            camera_images: Multi-view images
                - Shape (B, N_cam, 3, H, W): single frame
                - Shape (B, T, N_cam, 3, H, W): temporal sequence
            camera_intrinsics: Camera intrinsics
                - Shape (B, N_cam, 3, 3) or (B, T, N_cam, 3, 3)
            camera_extrinsics: Camera extrinsics
                - Shape (B, N_cam, 4, 4) or (B, T, N_cam, 4, 4)
            ego_transform: Ego-motion transform, shape (B, 4, 4)
            lidar_mask: Optional boolean mask
                - Shape (B, N) or (B, T, N) where True = valid point
            return_intermediate: Whether to return intermediate features

        Returns:
            Dictionary containing:
            - 'cls_scores': Classification scores
            - 'bbox_preds': Bounding box predictions
            - (optional) 'intermediate' dict if return_intermediate=True
        """
        from .utils.core import normalize_lidar_points_and_mask

        # Detect temporal dimension in inputs
        # Camera images: 5D = (B, N_cam, C, H, W), 6D = (B, T, N_cam, C, H, W)
        has_temporal_camera = camera_images.dim() == 6
        has_temporal_lidar = lidar_points.dim() == 4

        # If we have temporal inputs with T > 1, delegate to forward_sequence
        if has_temporal_camera and camera_images.shape[1] > 1:
            # Multi-frame temporal sequence - use forward_sequence
            return self.forward_sequence(
                lidar_points_seq=lidar_points,
                camera_images_seq=camera_images,
                camera_intrinsics_seq=camera_intrinsics,
                camera_extrinsics_seq=camera_extrinsics,
                ego_pose_seq=ego_transform.unsqueeze(1) if ego_transform is not None else None,
                lidar_mask_seq=lidar_mask,
                return_intermediate=return_intermediate
            )

        # Handle T=1 temporal inputs by squeezing
        if has_temporal_camera and camera_images.shape[1] == 1:
            camera_images = camera_images[:, 0]  # (B, N_cam, C, H, W)
            camera_intrinsics = camera_intrinsics[:, 0] if camera_intrinsics.dim() == 5 else camera_intrinsics
            camera_extrinsics = camera_extrinsics[:, 0] if camera_extrinsics.dim() == 5 else camera_extrinsics

        # Normalize LiDAR points and mask using the authoritative helper
        # This handles 4D->3D conversion and mask inference/validation
        lidar_points_3d, lidar_mask_2d = normalize_lidar_points_and_mask(
            lidar_points, lidar_mask, squeeze_temporal=True
        )

        intermediate = {}

        # 1. LiDAR BEV Encoding (with normalized mask)
        lidar_bev = self.lidar_encoder(lidar_points_3d, mask=lidar_mask_2d)
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
    ) -> List[List[Box3D]]:
        """
        Run inference and return decoded detections for all samples in batch.

        Args:
            lidar_points: LiDAR point cloud, shape (B, max_points, 4)
            camera_images: Multi-view images, shape (B, N_cams, 3, H, W)
            camera_intrinsics: Camera intrinsics, shape (B, N_cams, 3, 3)
            camera_extrinsics: Camera extrinsics, shape (B, N_cams, 4, 4)
            ego_transform: Ego-motion transform, shape (B, 4, 4)

        Returns:
            List of List[Box3D] detections after NMS, one list per sample in batch.
            For batch_size=1, returns [[detections_for_sample_0]].
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

            # Post-process (returns List[List[Box3D]] - one per sample)
            detections_batch = self.post_processor(
                outputs['cls_scores'],
                outputs['bbox_preds']
            )

        # Return all detections for all samples in batch
        return detections_batch
    
    def reset_temporal_state(self):
        """Reset temporal state (e.g., at scene boundaries or start of new batch).

        CRITICAL (P0-1): This MUST be called at the start of each batch when using
        forward_sequence to prevent cross-batch state leakage.
        """
        if self.use_temporal_attention:
            self.temporal_attention.reset()

        if self.use_mc_convrnn:
            self.mc_hidden_state = None
            if hasattr(self, '_prev_features'):
                delattr(self, '_prev_features')

    def forward_sequence(
        self,
        lidar_points_seq: torch.Tensor,
        camera_images_seq: torch.Tensor,
        camera_intrinsics_seq: torch.Tensor,
        camera_extrinsics_seq: torch.Tensor,
        ego_pose_seq: torch.Tensor,
        lidar_mask_seq: Optional[torch.Tensor] = None,
        scene_tokens: Optional[List[List[str]]] = None,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with IN-SAMPLE temporal unrolling for fusion training.

        CRITICAL FIX (P0-1): This method performs temporal unrolling WITHIN a batch,
        preventing cross-batch state leakage that occurs when using stateful temporal
        modules with shuffled DataLoader.

        The temporal state is reset at the start of this method and evolves only
        within the sequence. This ensures:
        1. No information leaks from previous batches
        2. Temporal modules receive coherent sequences (not random shuffled frames)
        3. Gradients flow correctly through the temporal unroll

        Args:
            lidar_points_seq: LiDAR sequences, shape (B, T, N_pts, 4)
            camera_images_seq: Camera image sequences, shape (B, T, N_cam, 3, H, W)
            camera_intrinsics_seq: Intrinsics sequences, shape (B, T, N_cam, 3, 3)
            camera_extrinsics_seq: Extrinsics sequences, shape (B, T, N_cam, 4, 4)
            ego_pose_seq: Ego pose sequences, shape (B, T, 4, 4)
            lidar_mask_seq: Optional boolean mask (B, T, N_pts), True = valid point
            scene_tokens: Optional scene tokens (B, T) for scene-boundary resets
            return_intermediate: Whether to return intermediate features

        Returns:
            Dictionary containing detection outputs for the LAST timestep:
            - 'cls_scores': Classification scores, shape (B, num_classes, H, W)
            - 'bbox_preds': Bounding box predictions, shape (B, 7, H, W)
            - (optional) 'intermediate_seq': list of intermediate feature dicts per timestep
        """
        B, T = lidar_points_seq.shape[:2]
        device = lidar_points_seq.device

        # CRITICAL: Reset temporal state at start of each batch
        self.reset_temporal_state()

        intermediate_seq = [] if return_intermediate else None

        # Track previous scene tokens for scene-boundary detection
        prev_scene_tokens = None

        # Unroll through time steps WITHIN this batch
        for t in range(T):
            # Scene-boundary reset: if scene changes mid-sequence, reset temporal state
            if scene_tokens is not None and t > 0:
                curr_tokens = scene_tokens[t] if isinstance(scene_tokens, list) else None
                if curr_tokens is not None and prev_scene_tokens is not None:
                    # Check each sample for scene boundary
                    for b in range(B):
                        if curr_tokens[b] != prev_scene_tokens[b]:
                            # Scene boundary detected for sample b
                            # Reset temporal state for this sample
                            self._reset_temporal_state_sample(b)
                prev_scene_tokens = curr_tokens

            # Extract current timestep data
            lidar_t = lidar_points_seq[:, t]  # (B, N_pts, 4)
            camera_t = camera_images_seq[:, t]  # (B, N_cam, 3, H, W)
            intrinsics_t = camera_intrinsics_seq[:, t]  # (B, N_cam, 3, 3)
            extrinsics_t = camera_extrinsics_seq[:, t]  # (B, N_cam, 4, 4)

            # Extract LiDAR mask for this timestep if provided
            lidar_mask_t = None
            if lidar_mask_seq is not None:
                lidar_mask_t = lidar_mask_seq[:, t]  # (B, N_pts)
                # Convert to bool if not already
                if lidar_mask_t.dtype != torch.bool:
                    lidar_mask_t = lidar_mask_t > 0.5

            # Compute ego transform from previous to current frame
            if t == 0:
                # First frame: identity transform (no previous frame)
                ego_transform = torch.eye(4, device=device).unsqueeze(0).expand(B, -1, -1)
            else:
                # Transform from frame t-1 to frame t
                # ego_pose is ego_to_world, so transform is:
                # current_to_world^-1 @ prev_to_world
                prev_pose = ego_pose_seq[:, t - 1]  # (B, 4, 4)
                curr_pose = ego_pose_seq[:, t]  # (B, 4, 4)
                ego_transform = torch.linalg.solve(curr_pose, prev_pose)  # More stable than inverse

            # Forward pass for this timestep (temporal state accumulates internally)
            outputs = self.forward(
                lidar_points=lidar_t,
                camera_images=camera_t,
                camera_intrinsics=intrinsics_t,
                camera_extrinsics=extrinsics_t,
                ego_transform=ego_transform,
                lidar_mask=lidar_mask_t,
                return_intermediate=return_intermediate
            )

            if return_intermediate:
                intermediate_seq.append(outputs.get('intermediate', {}))

        # Return outputs from the LAST timestep (for loss computation)
        result = {
            'cls_scores': outputs['cls_scores'],
            'bbox_preds': outputs['bbox_preds']
        }

        if return_intermediate:
            result['intermediate_seq'] = intermediate_seq

        return result

    def _reset_temporal_state_sample(self, sample_idx: int):
        """Reset temporal state for a single sample (for scene-boundary handling).

        This is a per-sample reset, used when a scene boundary is detected
        within a batch for a specific sample.

        Args:
            sample_idx: Index of the sample to reset
        """
        # For MemoryBank: reset history for this sample
        if self.use_temporal_attention and hasattr(self.temporal_attention, 'reset_sample'):
            self.temporal_attention.reset_sample(sample_idx)

        # For MCConvRNN: zero out hidden state for this sample
        if self.use_mc_convrnn:
            if self.mc_hidden_state is not None:
                self.mc_hidden_state[sample_idx] = 0.0
            if hasattr(self, '_prev_features') and self._prev_features is not None:
                self._prev_features[sample_idx] = 0.0
    
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
        fusion_type=config.get('fusion_type', 'cross_attention'),
        fusion_window_size=config.get('fusion_window_size', 7),
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
