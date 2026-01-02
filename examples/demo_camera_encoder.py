"""
Demo script for Camera BEV Encoder (BEVFormer).

This script demonstrates how to use the CameraBEVEncoder to convert
multi-view camera images into BEV features.
"""

import torch
import numpy as np
from modules.camera_encoder import CameraBEVEncoder


def create_synthetic_camera_data(batch_size=2, n_cam=6):
    """
    Create synthetic camera data for demonstration.
    
    Args:
        batch_size: Number of samples in batch
        n_cam: Number of cameras (default 6 for nuScenes)
    
    Returns:
        images: (B, N_cam, 3, H, W) - Camera images
        intrinsics: (B, N_cam, 3, 3) - Camera intrinsic matrices
        extrinsics: (B, N_cam, 4, 4) - Camera extrinsic matrices
    """
    img_h, img_w = 900, 1600
    
    # Generate random images
    images = torch.randn(batch_size, n_cam, 3, img_h, img_w)
    
    # Create camera intrinsics (typical for nuScenes)
    intrinsics = torch.zeros(batch_size, n_cam, 3, 3)
    for b in range(batch_size):
        for cam in range(n_cam):
            # Focal lengths and principal point
            fx, fy = 1000.0, 1000.0
            cx, cy = 800.0, 450.0
            
            intrinsics[b, cam] = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
    
    # Create camera extrinsics (ego to camera transform)
    # Simulate 6 cameras around the vehicle
    extrinsics = torch.zeros(batch_size, n_cam, 4, 4)
    
    # Camera positions (simplified)
    # Front, Front-Left, Front-Right, Back, Back-Left, Back-Right
    camera_yaws = [0, np.pi/6, -np.pi/6, np.pi, 5*np.pi/6, -5*np.pi/6]
    camera_positions = [
        [1.5, 0.0, 1.5],    # Front
        [1.0, 0.5, 1.5],    # Front-Left
        [1.0, -0.5, 1.5],   # Front-Right
        [-1.5, 0.0, 1.5],   # Back
        [-1.0, 0.5, 1.5],   # Back-Left
        [-1.0, -0.5, 1.5],  # Back-Right
    ]
    
    for b in range(batch_size):
        for cam in range(n_cam):
            yaw = camera_yaws[cam]
            pos = camera_positions[cam]
            
            # Rotation matrix (yaw only)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            rotation = torch.tensor([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ], dtype=torch.float32)
            
            # Create 4x4 transformation matrix
            extrinsics[b, cam, :3, :3] = rotation
            extrinsics[b, cam, :3, 3] = torch.tensor(pos)
            extrinsics[b, cam, 3, 3] = 1.0
    
    return images, intrinsics, extrinsics


def main():
    """Main demo function."""
    print("=" * 80)
    print("Camera BEV Encoder (BEVFormer) Demo")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    n_cam = 6
    bev_h, bev_w = 200, 200
    embed_dim = 256
    num_layers = 6
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of cameras: {n_cam}")
    print(f"  BEV grid size: {bev_h} x {bev_w}")
    print(f"  Feature dimension: {embed_dim}")
    print(f"  Number of attention layers: {num_layers}")
    
    # Create encoder
    print("\n" + "-" * 80)
    print("Creating Camera BEV Encoder...")
    encoder = CameraBEVEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dim=embed_dim,
        num_layers=num_layers,
        bev_x_range=(-51.2, 51.2),
        bev_y_range=(-51.2, 51.2),
        bev_z_ref=0.0
    )
    encoder.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create synthetic data
    print("\n" + "-" * 80)
    print("Creating synthetic camera data...")
    images, intrinsics, extrinsics = create_synthetic_camera_data(
        batch_size=batch_size,
        n_cam=n_cam
    )
    
    print(f"  Images shape: {images.shape}")
    print(f"  Intrinsics shape: {intrinsics.shape}")
    print(f"  Extrinsics shape: {extrinsics.shape}")
    
    # Forward pass
    print("\n" + "-" * 80)
    print("Running forward pass...")
    
    with torch.no_grad():
        bev_features = encoder(images, intrinsics, extrinsics)
    
    print(f"  Output BEV features shape: {bev_features.shape}")
    print(f"  Output dtype: {bev_features.dtype}")
    print(f"  Output device: {bev_features.device}")
    
    # Statistics
    print("\n" + "-" * 80)
    print("Output statistics:")
    print(f"  Mean: {bev_features.mean().item():.6f}")
    print(f"  Std: {bev_features.std().item():.6f}")
    print(f"  Min: {bev_features.min().item():.6f}")
    print(f"  Max: {bev_features.max().item():.6f}")
    print(f"  All finite: {torch.isfinite(bev_features).all()}")
    
    # Visualize BEV grid coordinates
    print("\n" + "-" * 80)
    print("BEV grid coordinates:")
    bev_coords = encoder.bev_coords
    print(f"  Shape: {bev_coords.shape}")
    print(f"  X range: [{bev_coords[:, :, 0].min():.2f}, {bev_coords[:, :, 0].max():.2f}]")
    print(f"  Y range: [{bev_coords[:, :, 1].min():.2f}, {bev_coords[:, :, 1].max():.2f}]")
    print(f"  Z value: {bev_coords[0, 0, 2].item():.2f}")
    
    # Component information
    print("\n" + "-" * 80)
    print("Encoder components:")
    print(f"  Backbone: {encoder.backbone.__class__.__name__}")
    print(f"  Neck: {encoder.neck.__class__.__name__}")
    print(f"  Cross-attention layers: {len(encoder.cross_attention_layers)}")
    print(f"  BEV queries shape: {encoder.bev_queries.shape}")
    print(f"  BEV positional encoding shape: {encoder.bev_pos_embed.shape}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
