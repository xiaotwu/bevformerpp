"""
Demo script for Spatial Fusion Module.

This script demonstrates how to use the spatial fusion module to combine
LiDAR and camera BEV features.
"""

import torch
import numpy as np
from modules.fusion import SpatialFusionModule
from modules.lidar_encoder import LiDARBEVEncoder
from modules.camera_encoder import CameraBEVEncoder
from modules.data_structures import BEVGridConfig


def demo_fusion_module():
    """
    Demonstrate the spatial fusion module with synthetic data.
    """
    print("=" * 80)
    print("Spatial Fusion Module Demo")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    bev_h, bev_w = 200, 200
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  BEV dimensions: {bev_h} x {bev_w}")
    print(f"  LiDAR channels: {lidar_channels}")
    print(f"  Camera channels: {camera_channels}")
    print(f"  Fused channels: {fused_channels}")
    
    # Initialize fusion module
    print("\n1. Initializing Spatial Fusion Module...")
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion_module.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in fusion_module.parameters())
    print(f"   Number of parameters: {num_params:,}")
    
    # Generate synthetic features
    print("\n2. Generating synthetic BEV features...")
    F_lidar = torch.randn(batch_size, lidar_channels, bev_h, bev_w)
    F_cam = torch.randn(batch_size, camera_channels, bev_h, bev_w)
    
    print(f"   LiDAR features shape: {F_lidar.shape}")
    print(f"   Camera features shape: {F_cam.shape}")
    
    # Verify alignment
    print("\n3. Verifying BEV grid alignment...")
    is_aligned = fusion_module.verify_alignment(F_lidar, F_cam)
    print(f"   Alignment check: {'✓ PASSED' if is_aligned else '✗ FAILED'}")
    
    # Perform fusion
    print("\n4. Performing spatial fusion...")
    with torch.no_grad():
        F_fused = fusion_module(F_lidar, F_cam)
    
    print(f"   Fused features shape: {F_fused.shape}")
    print(f"   Output is finite: {torch.isfinite(F_fused).all()}")
    print(f"   Output mean: {F_fused.mean().item():.4f}")
    print(f"   Output std: {F_fused.std().item():.4f}")
    print(f"   Output min: {F_fused.min().item():.4f}")
    print(f"   Output max: {F_fused.max().item():.4f}")
    
    # Test with misaligned features
    print("\n5. Testing alignment verification with misaligned features...")
    F_cam_wrong = torch.randn(batch_size, camera_channels, bev_h - 10, bev_w)
    is_aligned_wrong = fusion_module.verify_alignment(F_lidar, F_cam_wrong)
    print(f"   Misaligned features detected: {'✓ PASSED' if not is_aligned_wrong else '✗ FAILED'}")
    
    print("\n" + "=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)


def demo_full_pipeline():
    """
    Demonstrate the full pipeline from raw data to fused features.
    """
    print("\n" + "=" * 80)
    print("Full Pipeline Demo (LiDAR + Camera → Fusion)")
    print("=" * 80)
    
    # Configuration
    batch_size = 1
    n_cam = 6
    img_h, img_w = 900, 1600
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of cameras: {n_cam}")
    print(f"  Image size: {img_h} x {img_w}")
    
    # Initialize BEV grid config
    config = BEVGridConfig()
    bev_h, bev_w = config.grid_size
    print(f"  BEV grid size: {bev_h} x {bev_w}")
    
    # Initialize encoders
    print("\n1. Initializing encoders...")
    lidar_encoder = LiDARBEVEncoder(config, out_channels=64)
    camera_encoder = CameraBEVEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dim=256,
        num_layers=2  # Fewer layers for demo
    )
    fusion_module = SpatialFusionModule(
        lidar_channels=64,
        camera_channels=256,
        fused_channels=256
    )
    
    # Set to eval mode
    lidar_encoder.eval()
    camera_encoder.eval()
    fusion_module.eval()
    
    print("   ✓ LiDAR encoder initialized")
    print("   ✓ Camera encoder initialized")
    print("   ✓ Fusion module initialized")
    
    # Generate synthetic data
    print("\n2. Generating synthetic sensor data...")
    
    # LiDAR point cloud (1000 points)
    n_points = 1000
    points = np.random.randn(n_points, 4).astype(np.float32)
    points[:, 0] = points[:, 0] * 20  # x: -40 to 40
    points[:, 1] = points[:, 1] * 20  # y: -40 to 40
    points[:, 2] = points[:, 2] * 2   # z: -4 to 4
    points[:, 3] = np.abs(points[:, 3])  # intensity: positive
    points_batch = [points]
    
    print(f"   LiDAR point cloud: {n_points} points")
    
    # Camera images
    images = torch.randn(batch_size, n_cam, 3, img_h, img_w)
    
    # Camera parameters
    intrinsics = torch.zeros(batch_size, n_cam, 3, 3)
    extrinsics = torch.zeros(batch_size, n_cam, 4, 4)
    
    for cam in range(n_cam):
        intrinsics[0, cam] = torch.tensor([
            [1000.0, 0, 800.0],
            [0, 1000.0, 450.0],
            [0, 0, 1.0]
        ])
        extrinsics[0, cam] = torch.eye(4)
        extrinsics[0, cam, 0, 3] = cam * 0.5  # Small offset per camera
    
    print(f"   Camera images: {n_cam} views")
    
    # Forward pass through encoders
    print("\n3. Encoding sensor data to BEV...")
    
    with torch.no_grad():
        print("   Processing LiDAR...")
        F_lidar = lidar_encoder(points_batch)
        print(f"   ✓ LiDAR BEV features: {F_lidar.shape}")
        
        print("   Processing cameras...")
        F_cam = camera_encoder(images, intrinsics, extrinsics)
        print(f"   ✓ Camera BEV features: {F_cam.shape}")
    
    # Fusion
    print("\n4. Fusing LiDAR and camera features...")
    with torch.no_grad():
        F_fused = fusion_module(F_lidar, F_cam)
    
    print(f"   ✓ Fused BEV features: {F_fused.shape}")
    print(f"   Output statistics:")
    print(f"     Mean: {F_fused.mean().item():.4f}")
    print(f"     Std:  {F_fused.std().item():.4f}")
    print(f"     Min:  {F_fused.min().item():.4f}")
    print(f"     Max:  {F_fused.max().item():.4f}")
    
    print("\n" + "=" * 80)
    print("Full pipeline demo completed successfully!")
    print("=" * 80)
    
    return F_fused


def demo_memory_usage():
    """
    Demonstrate memory usage for different BEV grid sizes.
    """
    print("\n" + "=" * 80)
    print("Memory Usage Analysis")
    print("=" * 80)
    
    fusion_module = SpatialFusionModule(
        lidar_channels=64,
        camera_channels=256,
        fused_channels=256,
        num_heads=8
    )
    
    # Test different grid sizes
    grid_sizes = [(50, 50), (100, 100), (200, 200)]
    
    print("\nTesting different BEV grid sizes:")
    print(f"{'Grid Size':<15} {'Attention Matrix':<20} {'Memory (approx)':<15}")
    print("-" * 50)
    
    for h, w in grid_sizes:
        spatial_size = h * w
        attn_matrix_size = spatial_size * spatial_size
        # Approximate memory: attention matrix in float32
        memory_mb = (attn_matrix_size * 4) / (1024 * 1024)
        
        print(f"{h}x{w:<12} {spatial_size}x{spatial_size:<15} {memory_mb:.1f} MB")
    
    print("\nNote: Full attention has O(N²) memory complexity where N = H*W")
    print("For large grids, consider using deformable attention or sparse attention.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demos
    demo_fusion_module()
    
    print("\n" + "=" * 80)
    input("Press Enter to continue to full pipeline demo...")
    
    demo_full_pipeline()
    
    print("\n" + "=" * 80)
    input("Press Enter to continue to memory analysis...")
    
    demo_memory_usage()
    
    print("\n✓ All demos completed!")
