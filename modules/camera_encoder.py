"""
Camera BEV Encoder (BEVFormer) Implementation

This module implements the camera-based BEV feature extraction using:
1. Image backbone (ResNet50) for feature extraction
2. FPN neck for multi-scale features
3. Spatial cross-attention to project image features to BEV space
4. Camera projection utilities for geometric transformations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from .backbone import ResNetBackbone
from .neck import FPN
from .attention import SpatialCrossAttention


def project_bev_to_image(bev_coords, intrinsics, extrinsics, bev_config):
    """
    Project BEV grid coordinates to image space.
    
    Args:
        bev_coords: (H, W, 3) - 3D coordinates of BEV grid points in ego frame
        intrinsics: (B, N_cam, 3, 3) - Camera intrinsic matrices
        extrinsics: (B, N_cam, 4, 4) - Camera extrinsic matrices (ego to camera)
        bev_config: dict with 'x_min', 'x_max', 'y_min', 'y_max', 'z_ref'
    
    Returns:
        reference_points: (B, N_cam, H*W, 2) - Normalized image coordinates [0, 1]
        valid_mask: (B, N_cam, H*W) - Boolean mask for valid projections
    """
    B, N_cam = intrinsics.shape[:2]
    H, W = bev_coords.shape[:2]
    device = intrinsics.device
    
    # Flatten BEV coordinates: (H*W, 3)
    bev_coords_flat = bev_coords.reshape(-1, 3)  # (H*W, 3)
    
    # Add homogeneous coordinate: (H*W, 4)
    bev_coords_homo = torch.cat([
        bev_coords_flat,
        torch.ones(H*W, 1, device=device)
    ], dim=-1)  # (H*W, 4)
    
    # Initialize outputs
    reference_points = torch.zeros(B, N_cam, H*W, 2, device=device)
    valid_mask = torch.zeros(B, N_cam, H*W, dtype=torch.bool, device=device)
    
    for b in range(B):
        for cam in range(N_cam):
            # Transform from ego to camera frame
            # extrinsic: (4, 4) transforms ego -> camera
            cam_coords_homo = torch.matmul(
                extrinsics[b, cam],  # (4, 4)
                bev_coords_homo.T    # (4, H*W)
            )  # (4, H*W)
            
            cam_coords = cam_coords_homo[:3, :]  # (3, H*W)
            
            # Check if points are in front of camera (positive z)
            valid_depth = cam_coords[2, :] > 0.1  # (H*W,)
            
            # Project to image plane
            # K @ [X, Y, Z]^T = [u*Z, v*Z, Z]^T
            img_coords_homo = torch.matmul(
                intrinsics[b, cam],  # (3, 3)
                cam_coords           # (3, H*W)
            )  # (3, H*W)
            
            # Normalize by depth
            img_coords = img_coords_homo[:2, :] / (img_coords_homo[2:3, :] + 1e-6)  # (2, H*W)
            
            # Normalize to [0, 1] range (assuming image size is known)
            # For grid_sample, we need [-1, 1] range
            # Let's assume image size is 900x1600 (nuScenes default)
            img_h, img_w = 900, 1600
            
            # Convert to [-1, 1] for grid_sample
            img_coords_norm = torch.stack([
                2.0 * img_coords[0, :] / img_w - 1.0,  # u
                2.0 * img_coords[1, :] / img_h - 1.0   # v
            ], dim=0)  # (2, H*W)
            
            # Check if points are within image bounds
            valid_u = (img_coords_norm[0, :] >= -1.0) & (img_coords_norm[0, :] <= 1.0)
            valid_v = (img_coords_norm[1, :] >= -1.0) & (img_coords_norm[1, :] <= 1.0)
            valid_proj = valid_u & valid_v & valid_depth
            
            # Store results
            reference_points[b, cam, :, :] = img_coords_norm.T  # (H*W, 2)
            valid_mask[b, cam, :] = valid_proj
    
    return reference_points, valid_mask


def backproject_2d_to_3d(points_2d, depth, intrinsics, extrinsics):
    """
    Back-project 2D image points to 3D ego frame.
    
    Args:
        points_2d: (N, 2) - Image coordinates (u, v) in pixels
        depth: (N,) - Depth values in meters
        intrinsics: (3, 3) - Camera intrinsic matrix
        extrinsics: (4, 4) - Camera extrinsic matrix (ego to camera)
    
    Returns:
        points_3d: (N, 3) - 3D points in ego frame
    """
    N = points_2d.shape[0]
    device = points_2d.device
    
    # Create homogeneous image coordinates
    points_2d_homo = torch.cat([
        points_2d,
        torch.ones(N, 1, device=device)
    ], dim=-1)  # (N, 3)
    
    # Inverse intrinsics
    K_inv = torch.inverse(intrinsics)  # (3, 3)
    
    # Back-project to camera frame
    cam_coords = torch.matmul(K_inv, points_2d_homo.T)  # (3, N)
    cam_coords = cam_coords * depth.unsqueeze(0)  # (3, N) * (1, N)
    
    # Add homogeneous coordinate
    cam_coords_homo = torch.cat([
        cam_coords,
        torch.ones(1, N, device=device)
    ], dim=0)  # (4, N)
    
    # Transform to ego frame (inverse of extrinsic)
    T_cam_to_ego = torch.inverse(extrinsics)  # (4, 4)
    ego_coords_homo = torch.matmul(T_cam_to_ego, cam_coords_homo)  # (4, N)
    
    points_3d = ego_coords_homo[:3, :].T  # (N, 3)
    
    return points_3d


class CameraBEVEncoder(nn.Module):
    """
    Camera BEV Encoder using BEVFormer architecture.
    
    Converts multi-view camera images to BEV features through:
    1. Image feature extraction (ResNet50 + FPN)
    2. Spatial cross-attention (image features -> BEV queries)
    3. Learnable BEV queries with positional encoding
    """
    
    def __init__(
        self,
        bev_h=200,
        bev_w=200,
        bev_z_ref=0.0,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        bev_x_range=(-51.2, 51.2),
        bev_y_range=(-51.2, 51.2),
        img_h=900,
        img_w=1600
    ):
        """
        Args:
            bev_h: BEV grid height
            bev_w: BEV grid width
            bev_z_ref: Reference height for BEV plane (meters)
            embed_dim: Feature embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            bev_x_range: (x_min, x_max) in meters
            bev_y_range: (y_min, y_max) in meters
            img_h: Input image height
            img_w: Input image width
        """
        super().__init__()
        
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_z_ref = bev_z_ref
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.img_h = img_h
        self.img_w = img_w
        
        # BEV configuration
        self.bev_config = {
            'x_min': bev_x_range[0],
            'x_max': bev_x_range[1],
            'y_min': bev_y_range[0],
            'y_max': bev_y_range[1],
            'z_ref': bev_z_ref
        }
        
        # Image backbone and neck
        self.backbone = ResNetBackbone(
            model_name='resnet50',
            pretrained=True,
            out_indices=(1, 2, 3, 4)
        )
        self.neck = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=embed_dim
        )
        
        # Learnable BEV queries
        self.bev_queries = nn.Parameter(
            torch.randn(bev_h * bev_w, embed_dim)
        )
        
        # Positional encoding for BEV queries
        self.bev_pos_embed = nn.Parameter(
            torch.randn(bev_h * bev_w, embed_dim)
        )
        
        # Spatial cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            SpatialCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
        # FFN (Feed-Forward Network) after each attention layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(0.1)
            )
            for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
        # Initialize BEV grid coordinates
        self.register_buffer('bev_coords', self._create_bev_coords())
    
    def _create_bev_coords(self):
        """
        Create 3D coordinates for BEV grid points in ego frame.
        
        Returns:
            bev_coords: (H, W, 3) - [x, y, z] coordinates
        """
        x_min, x_max = self.bev_config['x_min'], self.bev_config['x_max']
        y_min, y_max = self.bev_config['y_min'], self.bev_config['y_max']
        z_ref = self.bev_config['z_ref']
        
        # Create grid
        x = torch.linspace(x_min, x_max, self.bev_w)
        y = torch.linspace(y_min, y_max, self.bev_h)
        
        # Note: In BEV, typically x is forward, y is left
        # Grid indexing: [H, W] corresponds to [y, x]
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        zz = torch.full_like(xx, z_ref)
        
        bev_coords = torch.stack([xx, yy, zz], dim=-1)  # (H, W, 3)
        
        return bev_coords
    
    def forward(self, images, intrinsics, extrinsics):
        """
        Forward pass of camera BEV encoder.
        
        Args:
            images: (B, N_cam, 3, H, W) - Multi-view images
            intrinsics: (B, N_cam, 3, 3) - Camera intrinsic matrices
            extrinsics: (B, N_cam, 4, 4) - Camera extrinsic matrices (ego to camera)
        
        Returns:
            bev_features: (B, C, H_bev, W_bev) - BEV feature map
        """
        B, N_cam, C, H, W = images.shape
        device = images.device
        
        # 1. Extract image features
        # Reshape for batch processing: (B*N_cam, 3, H, W)
        images_flat = images.reshape(B * N_cam, C, H, W)
        
        # Backbone features (multi-scale)
        backbone_feats = self.backbone(images_flat)
        
        # FPN neck (multi-scale fusion)
        fpn_feats = self.neck(backbone_feats)
        
        # Use the finest scale feature map for attention
        # fpn_feats[-1]: (B*N_cam, embed_dim, H_feat, W_feat)
        img_features = fpn_feats[-1]
        
        # 2. Project BEV grid to image space
        # reference_points: (B, N_cam, H_bev*W_bev, 2)
        # valid_mask: (B, N_cam, H_bev*W_bev)
        reference_points, valid_mask = project_bev_to_image(
            self.bev_coords,
            intrinsics,
            extrinsics,
            self.bev_config
        )
        
        # 3. Initialize BEV queries
        # (B, H_bev*W_bev, embed_dim)
        bev_queries = self.bev_queries.unsqueeze(0).repeat(B, 1, 1)
        bev_pos = self.bev_pos_embed.unsqueeze(0).repeat(B, 1, 1)
        
        # 4. Apply spatial cross-attention layers
        bev_embed = bev_queries
        
        for i in range(self.num_layers):
            # Cross-attention
            bev_embed_attn = self.cross_attention_layers[i](
                query=bev_embed,
                key=None,
                value=img_features,
                query_pos=bev_pos,
                reference_points_cam=reference_points,
                bev_mask=valid_mask,
                spatial_shapes=None,
                level_start_index=None
            )
            
            # Residual connection + layer norm
            bev_embed = self.layer_norms[i](bev_embed + bev_embed_attn)
            
            # FFN
            bev_embed_ffn = self.ffns[i](bev_embed)
            
            # Residual connection + layer norm
            bev_embed = self.ffn_norms[i](bev_embed + bev_embed_ffn)
        
        # 5. Reshape to BEV grid
        # (B, H_bev*W_bev, embed_dim) -> (B, embed_dim, H_bev, W_bev)
        bev_features = bev_embed.permute(0, 2, 1).reshape(
            B, self.embed_dim, self.bev_h, self.bev_w
        )
        
        return bev_features


# Standalone projection utilities for testing
def project_3d_to_2d(points_3d, intrinsics, extrinsics):
    """
    Project 3D points in ego frame to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) - 3D points in ego frame [x, y, z]
        intrinsics: (3, 3) - Camera intrinsic matrix
        extrinsics: (4, 4) - Camera extrinsic matrix (ego to camera)
    
    Returns:
        points_2d: (N, 2) - 2D image coordinates [u, v] in pixels
        valid_mask: (N,) - Boolean mask for valid projections
    """
    N = points_3d.shape[0]
    device = points_3d.device
    
    # Add homogeneous coordinate
    points_3d_homo = torch.cat([
        points_3d,
        torch.ones(N, 1, device=device)
    ], dim=-1)  # (N, 4)
    
    # Transform to camera frame
    cam_coords_homo = torch.matmul(
        extrinsics,  # (4, 4)
        points_3d_homo.T  # (4, N)
    )  # (4, N)
    
    cam_coords = cam_coords_homo[:3, :]  # (3, N)
    
    # Check valid depth
    valid_depth = cam_coords[2, :] > 0.1  # (N,)
    
    # Project to image plane
    img_coords_homo = torch.matmul(
        intrinsics,  # (3, 3)
        cam_coords   # (3, N)
    )  # (3, N)
    
    # Normalize by depth
    points_2d = img_coords_homo[:2, :] / (img_coords_homo[2:3, :] + 1e-6)  # (2, N)
    points_2d = points_2d.T  # (N, 2)
    
    # Check if within image bounds (assuming 900x1600)
    valid_u = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < 1600)
    valid_v = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < 900)
    valid_mask = valid_u & valid_v & valid_depth
    
    return points_2d, valid_mask
