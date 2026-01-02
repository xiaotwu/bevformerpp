import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal
from .backbone import ResNetBackbone
from .neck import FPN
from .attention import SpatialCrossAttention, TemporalSelfAttention
from .convrnn import ConvGRU
from .memory_bank import MemoryBank
from .temporal_attention import TemporalAggregationModule
from .mc_convrnn import MCConvRNN
from .utils import warp_bev, align_bev_features


def project_bev_to_image(
    bev_positions: torch.Tensor,
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_size: tuple = (224, 400)
) -> tuple:
    """
    Project 3D BEV grid positions to 2D image coordinates for all cameras.

    Args:
        bev_positions: 3D positions of BEV grid centers, shape (B, H*W, 3)
                      where 3 = (x, y, z) in ego/world coordinates
        intrinsics: Camera intrinsic matrices, shape (B, N_cam, 3, 3)
        extrinsics: Camera extrinsic matrices (world-to-camera), shape (B, N_cam, 4, 4)
        image_size: Target image size (H, W) for normalization

    Returns:
        ref_points: Normalized 2D reference points, shape (B, N_cam, H*W, 2)
                   Values in range [-1, 1] for grid_sample
        valid_mask: Mask for valid projections, shape (B, N_cam, H*W)
                   True where point is in front of camera and within image bounds
    """
    B, L, _ = bev_positions.shape
    N_cam = intrinsics.shape[1]
    H_img, W_img = image_size
    device = bev_positions.device

    # Convert to homogeneous coordinates: (B, L, 4)
    ones = torch.ones(B, L, 1, device=device)
    bev_homo = torch.cat([bev_positions, ones], dim=-1)  # (B, L, 4)

    # Process each camera
    ref_points_list = []
    valid_mask_list = []

    for cam_idx in range(N_cam):
        # Get camera matrices for this camera
        K = intrinsics[:, cam_idx]  # (B, 3, 3)
        T = extrinsics[:, cam_idx]  # (B, 4, 4) - world/ego to camera transform

        # Transform points to camera frame: (B, L, 4) @ (B, 4, 4).T -> (B, L, 4)
        # Note: We need P_cam = T @ P_world, so we do (B, L, 4) @ (B, 4, 4).T
        points_cam = torch.bmm(bev_homo, T.transpose(1, 2))  # (B, L, 4)

        # Extract XYZ in camera frame
        X_cam = points_cam[:, :, 0]  # (B, L)
        Y_cam = points_cam[:, :, 1]  # (B, L)
        Z_cam = points_cam[:, :, 2]  # (B, L) - depth

        # Check if point is in front of camera (Z > 0)
        valid_depth = Z_cam > 0.1  # (B, L)

        # Avoid division by zero
        Z_cam_safe = Z_cam.clamp(min=0.1)

        # Project to normalized camera coordinates
        x_norm = X_cam / Z_cam_safe  # (B, L)
        y_norm = Y_cam / Z_cam_safe  # (B, L)

        # Stack for matrix multiplication: (B, L, 3)
        points_norm = torch.stack([x_norm, y_norm, torch.ones_like(x_norm)], dim=-1)

        # Apply intrinsics: (B, L, 3) @ (B, 3, 3).T -> (B, L, 3)
        points_img = torch.bmm(points_norm, K.transpose(1, 2))  # (B, L, 3)

        # Extract pixel coordinates
        u = points_img[:, :, 0]  # (B, L)
        v = points_img[:, :, 1]  # (B, L)

        # Check if within image bounds
        valid_u = (u >= 0) & (u < W_img)
        valid_v = (v >= 0) & (v < H_img)
        valid_bounds = valid_u & valid_v

        # Combined validity mask
        valid = valid_depth & valid_bounds  # (B, L)

        # Normalize to [-1, 1] for grid_sample
        u_normalized = 2.0 * (u / W_img) - 1.0  # [-1, 1]
        v_normalized = 2.0 * (v / H_img) - 1.0  # [-1, 1]

        # Clamp to valid range (for invalid points, clamp to border)
        u_normalized = u_normalized.clamp(-1, 1)
        v_normalized = v_normalized.clamp(-1, 1)

        # Stack: (B, L, 2)
        ref_points_cam = torch.stack([u_normalized, v_normalized], dim=-1)

        ref_points_list.append(ref_points_cam)
        valid_mask_list.append(valid)

    # Stack across cameras: (B, N_cam, L, 2) and (B, N_cam, L)
    ref_points = torch.stack(ref_points_list, dim=1)
    valid_mask = torch.stack(valid_mask_list, dim=1).float()

    return ref_points, valid_mask


class EnhancedBEVFormer(nn.Module):
    """
    Enhanced BEVFormer with configurable temporal fusion methods.
    
    Supports three temporal fusion strategies:
    - 'convgru': Simple ConvGRU baseline (original implementation)
    - 'temporal_attention': Transformer-based temporal attention with MemoryBank
    - 'mc_convrnn': Motion-Compensated ConvRNN with ego-motion warping,
                    dynamic motion fields, and visibility gating
    
    The temporal method can be selected at initialization time.
    """
    
    def __init__(self, bev_h=200, bev_w=200, embed_dim=256,
                 bev_x_range=(-51.2, 51.2), bev_y_range=(-51.2, 51.2),
                 bev_z_range=(-5.0, 3.0),
                 temporal_method: Literal['convgru', 'temporal_attention', 'mc_convrnn'] = 'convgru',
                 max_history: int = 5,
                 enable_bptt: bool = False,
                 # MC-ConvRNN ablation flags
                 mc_disable_warping: bool = False,
                 mc_disable_motion_field: bool = False,
                 mc_disable_visibility: bool = False):
        """
        Initialize EnhancedBEVFormer.
        
        Args:
            bev_h: BEV grid height
            bev_w: BEV grid width
            embed_dim: Feature embedding dimension
            bev_x_range: BEV x-axis range in meters
            bev_y_range: BEV y-axis range in meters
            bev_z_range: BEV z-axis range in meters
            temporal_method: Temporal fusion method:
                - 'convgru': Simple ConvGRU (baseline)
                - 'temporal_attention': Transformer attention with MemoryBank
                - 'mc_convrnn': Motion-Compensated ConvRNN (proposed)
            max_history: Maximum number of history frames for temporal_attention
            enable_bptt: Enable backpropagation through time for temporal modules
            mc_disable_warping: Ablation - disable ego-motion warping in MC-ConvRNN
            mc_disable_motion_field: Ablation - disable motion field estimation
            mc_disable_visibility: Ablation - disable visibility gating
        """
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.embed_dim = embed_dim
        self.temporal_method = temporal_method
        self.max_history = max_history

        # BEV grid configuration
        self.bev_x_range = bev_x_range
        self.bev_y_range = bev_y_range
        self.bev_z_range = bev_z_range
        self.bev_range = (bev_x_range[0], bev_x_range[1], bev_y_range[0], bev_y_range[1])

        # Components
        self.backbone = ResNetBackbone()
        self.neck = FPN(in_channels=self.backbone.out_channels, out_channels=embed_dim)

        self.spatial_cross_attention = SpatialCrossAttention(embed_dim=embed_dim)
        self.temporal_self_attention = TemporalSelfAttention(embed_dim=embed_dim)

        # Initialize temporal module based on selected method
        if temporal_method == 'convgru':
            self.conv_gru = ConvGRU(input_dim=embed_dim, hidden_dim=embed_dim)
            self.memory_bank = None  # Not used for ConvGRU
            self.temporal_attention = None
            self.mc_convrnn = None
        elif temporal_method == 'temporal_attention':
            self.conv_gru = None
            self.memory_bank = MemoryBank(max_length=max_history, enable_bptt=enable_bptt)
            self.temporal_attention = TemporalAggregationModule(
                embed_dim=embed_dim,
                num_heads=8,
                max_history=max_history,
                bev_range=self.bev_range,
                enable_bptt=enable_bptt
            )
            self.mc_convrnn = None
        elif temporal_method == 'mc_convrnn':
            self.conv_gru = None
            self.memory_bank = None
            self.temporal_attention = None
            self.mc_convrnn = MCConvRNN(
                input_channels=embed_dim,
                hidden_channels=embed_dim,
                bev_range=self.bev_range,
                disable_warping=mc_disable_warping,
                disable_motion_field=mc_disable_motion_field,
                disable_visibility=mc_disable_visibility
            )
            # State for MC-ConvRNN
            self._mc_hidden_state = None
            self._mc_prev_features = None
        else:
            raise ValueError(f"Unknown temporal_method: {temporal_method}. "
                           f"Must be one of: 'convgru', 'temporal_attention', 'mc_convrnn'")

        # BEV Queries (Learnable)
        self.bev_queries = nn.Parameter(torch.randn(bev_h * bev_w, embed_dim))

        # Initialize fixed BEV grid positions (3D coordinates in ego frame)
        # These represent the physical locations that each BEV cell corresponds to
        self._init_bev_grid()

    def _init_bev_grid(self):
        """Initialize the 3D positions for each BEV grid cell."""
        x_min, x_max = self.bev_x_range
        y_min, y_max = self.bev_y_range

        # Create grid of x, y positions
        x = torch.linspace(x_min, x_max, self.bev_w)
        y = torch.linspace(y_min, y_max, self.bev_h)

        # Create meshgrid (H, W)
        yy, xx = torch.meshgrid(y, x, indexing='ij')

        # Use ground level (z=0) as the reference height for projection
        # This ensures BEV points project to reasonable image locations
        # (objects are typically at or above ground level)
        z_ref = 0.0
        zz = torch.full_like(xx, z_ref)

        # Stack to get (H, W, 3) then flatten to (H*W, 3)
        bev_grid = torch.stack([xx, yy, zz], dim=-1)  # (H, W, 3)
        bev_grid = bev_grid.view(-1, 3)  # (H*W, 3)

        # Register as buffer (not a parameter, but moves with device)
        self.register_buffer('bev_grid', bev_grid)

    def forward(self, imgs, intrinsics, extrinsics, ego_pose, prev_bev=None, ego_motion=None):
        """
        Forward pass of EnhancedBEVFormer.

        Args:
            imgs: Multi-camera images, shape (B, N_cam, 3, H, W)
            intrinsics: Camera intrinsic matrices, shape (B, N_cam, 3, 3)
            extrinsics: Camera extrinsic matrices (ego-to-camera), shape (B, N_cam, 4, 4)
            ego_pose: Current ego pose (ego-to-world), shape (B, 4, 4)
            prev_bev: Previous BEV features, shape (B, C, H_bev, W_bev)
            ego_motion: Transform from previous to current ego frame, shape (B, 4, 4)

        Returns:
            curr_bev: Current BEV features, shape (B, C, H_bev, W_bev)
        """
        B, N_cam, C, H, W = imgs.shape

        # 1. Backbone + Neck
        imgs_flat = imgs.view(-1, C, H, W)
        feats = self.backbone(imgs_flat)
        # Use the last feature map from FPN for simplicity
        fpn_feats = self.neck(feats)[-1]  # (B*N_cam, embed_dim, H_feat, W_feat)
        _, _, H_feat, W_feat = fpn_feats.shape

        # 2. Compute Reference Points by projecting BEV grid to image planes
        # Expand BEV grid positions for batch: (H*W, 3) -> (B, H*W, 3)
        bev_positions = self.bev_grid.unsqueeze(0).expand(B, -1, -1)

        # GEOMETRY FIX: Scale intrinsics from image size to feature map size.
        # Intrinsics K are defined for input image size (H, W).
        # Feature maps are downsampled (H_feat, W_feat), so we need to scale K accordingly.
        # K_feat = S @ K where S is the downsampling scale matrix.
        scale_h = H_feat / H
        scale_w = W_feat / W
        intrinsics_feat = intrinsics.clone()
        intrinsics_feat[:, :, 0, 0] *= scale_w  # fx
        intrinsics_feat[:, :, 1, 1] *= scale_h  # fy
        intrinsics_feat[:, :, 0, 2] *= scale_w  # cx
        intrinsics_feat[:, :, 1, 2] *= scale_h  # cy

        # Project 3D BEV positions to 2D image coordinates using scaled intrinsics
        ref_points, bev_mask = project_bev_to_image(
            bev_positions,
            intrinsics_feat,  # Use intrinsics scaled for feature map
            extrinsics,
            image_size=(H_feat, W_feat)  # Feature map size for normalization
        )

        # Query: (B, L, C)
        query = self.bev_queries.unsqueeze(0).expand(B, -1, -1)

        bev_embed = self.spatial_cross_attention(
            query,
            value=fpn_feats,
            key=None,
            query_pos=None,
            reference_points_cam=ref_points,
            bev_mask=bev_mask,
            spatial_shapes=None,
            level_start_index=None
        )

        # bev_embed: (B, L, C) -> (B, C, H, W)
        bev_embed = bev_embed.permute(0, 2, 1).view(B, self.embed_dim, self.bev_h, self.bev_w)

        # 4. Temporal Fusion (based on selected method)
        if self.temporal_method == 'convgru':
            # Simple ConvGRU baseline
            if prev_bev is not None and ego_motion is not None:
                prev_bev_warped = warp_bev(prev_bev, ego_motion, bev_range=self.bev_range)
            else:
                prev_bev_warped = torch.zeros(B, self.embed_dim, self.bev_h, self.bev_w, device=imgs.device)
            curr_bev = self.conv_gru(bev_embed, prev_bev_warped)
            
        elif self.temporal_method == 'temporal_attention':
            # Transformer-based temporal attention with MemoryBank
            # TemporalAggregationModule handles memory bank internally
            curr_bev = self.temporal_attention(
                current_bev=bev_embed,
                ego_transform=ego_motion
            )
            
        elif self.temporal_method == 'mc_convrnn':
            # Motion-Compensated ConvRNN
            # Check if stored state has different batch size than current batch
            # This can happen at batch boundaries during evaluation (e.g., last batch is smaller)
            prev_features = self._mc_prev_features
            prev_hidden = self._mc_hidden_state

            if prev_features is not None and prev_features.shape[0] != B:
                # Batch size mismatch - reset temporal state
                prev_features = None
                prev_hidden = None

            if prev_hidden is not None and prev_hidden.shape[0] != B:
                # Also check hidden state batch size
                prev_hidden = None

            curr_bev, new_hidden = self.mc_convrnn(
                current_features=bev_embed,
                prev_features=prev_features,
                prev_hidden=prev_hidden,
                ego_motion=ego_motion
            )
            # Store state for next frame (detached for truncated BPTT)
            self._mc_prev_features = curr_bev.detach()
            self._mc_hidden_state = new_hidden.detach()
        
        return curr_bev
    
    def reset_temporal_state(self):
        """
        Reset temporal state at scene boundaries.
        
        IMPORTANT: Call this method when transitioning between scenes
        to prevent temporal state leakage.
        """
        if self.temporal_method == 'convgru':
            # ConvGRU state is managed via prev_bev argument, no internal state
            pass
        elif self.temporal_method == 'temporal_attention':
            if self.temporal_attention is not None:
                self.temporal_attention.reset()
        elif self.temporal_method == 'mc_convrnn':
            self._mc_hidden_state = None
            self._mc_prev_features = None

    def forward_sequence(self, seq_imgs, seq_intrinsics, seq_extrinsics, seq_ego_pose,
                          scene_tokens: Optional[list] = None):
        """
        Process a sequence of frames.
        
        Args:
            seq_imgs: (B, Seq, N_cam, 3, H, W)
            seq_intrinsics: (B, Seq, N_cam, 3, 3)
            seq_extrinsics: (B, Seq, N_cam, 4, 4)  
            seq_ego_pose: (B, Seq, 4, 4)
            scene_tokens: Optional list of scene tokens per timestep to detect scene boundaries.
                         If provided, temporal state is reset when scene changes.
        
        Returns:
            BEV features for each timestep: (B, Seq, C, H, W)
        """
        B, Seq, N_cam, C, H, W = seq_imgs.shape
        
        bev_outputs = []
        prev_bev = None
        prev_scene_token = None
        
        for t in range(Seq):
            # Check for scene boundary and reset temporal state if needed
            if scene_tokens is not None:
                current_scene = scene_tokens[t] if t < len(scene_tokens) else None
                if prev_scene_token is not None and current_scene != prev_scene_token:
                    self.reset_temporal_state()
                    prev_bev = None  # Also reset for ConvGRU
                prev_scene_token = current_scene
            
            imgs = seq_imgs[:, t]
            intrinsics = seq_intrinsics[:, t]
            extrinsics = seq_extrinsics[:, t]
            ego_pose = seq_ego_pose[:, t]
            
            ego_motion = None
            if t > 0 and prev_bev is not None:
                # Compute ego motion from t-1 to t
                # ego_pose is Ego -> Global (standard convention)
                # T_{t-1->t} = inv(T_{t}) * T_{t-1}
                pose_t = ego_pose
                pose_prev = seq_ego_pose[:, t-1]
                ego_motion = torch.matmul(torch.inverse(pose_t), pose_prev)
            
            curr_bev = self.forward(imgs, intrinsics, extrinsics, ego_pose, prev_bev, ego_motion)
            bev_outputs.append(curr_bev)
            prev_bev = curr_bev
            
        return torch.stack(bev_outputs, dim=1)  # (B, Seq, C, H, W)
