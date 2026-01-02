"""
Demonstration of LiDAR BEV Encoder.
Shows how to use the PointPillars-based encoder to convert point clouds to BEV features.
"""

import numpy as np
import torch
from modules.data_structures import BEVGridConfig
from modules.lidar_encoder import LiDARBEVEncoder


def main():
    print("=" * 80)
    print("LiDAR BEV Encoder Demonstration")
    print("=" * 80)
    
    # Create BEV grid configuration
    config = BEVGridConfig(
        x_min=-51.2, x_max=51.2,
        y_min=-51.2, y_max=51.2,
        z_min=-5.0, z_max=3.0,
        resolution=0.2
    )
    
    print(f"\nBEV Grid Configuration:")
    print(f"  X range: [{config.x_min}, {config.x_max}] meters")
    print(f"  Y range: [{config.y_min}, {config.y_max}] meters")
    print(f"  Z range: [{config.z_min}, {config.z_max}] meters")
    print(f"  Resolution: {config.resolution} meters/pixel")
    print(f"  Grid size: {config.grid_size} (H, W)")
    
    # Create encoder
    out_channels = 64
    encoder = LiDARBEVEncoder(config, out_channels=out_channels)
    encoder.eval()
    
    print(f"\nEncoder Configuration:")
    print(f"  Output channels: {out_channels}")
    print(f"  Total parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    # Generate synthetic point cloud
    print("\n" + "-" * 80)
    print("Generating synthetic point cloud...")
    
    n_points = 5000
    x = np.random.uniform(-50, 50, n_points)
    y = np.random.uniform(-50, 50, n_points)
    z = np.random.uniform(-4, 2, n_points)
    intensity = np.random.uniform(0, 1, n_points)
    
    points = np.stack([x, y, z, intensity], axis=1).astype(np.float32)
    
    print(f"  Generated {n_points} points")
    print(f"  Point cloud shape: {points.shape}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Create batch
    batch_size = 2
    points_batch = [points.copy() for _ in range(batch_size)]
    
    print(f"\n  Created batch of {batch_size} point clouds")
    
    # Forward pass
    print("\n" + "-" * 80)
    print("Running forward pass...")
    
    with torch.no_grad():
        bev_features = encoder(points_batch)
    
    print(f"  Output shape: {bev_features.shape}")
    print(f"  Output dtype: {bev_features.dtype}")
    print(f"  Output device: {bev_features.device}")
    
    # Analyze output
    print("\n" + "-" * 80)
    print("Output Analysis:")
    
    print(f"  Min value: {bev_features.min().item():.4f}")
    print(f"  Max value: {bev_features.max().item():.4f}")
    print(f"  Mean value: {bev_features.mean().item():.4f}")
    print(f"  Std value: {bev_features.std().item():.4f}")
    
    # Check for non-zero locations
    non_zero_mask = (bev_features != 0).any(dim=1)  # (B, H, W)
    non_zero_count = non_zero_mask.sum(dim=(1, 2))
    
    print(f"\n  Non-zero BEV cells per sample:")
    for i, count in enumerate(non_zero_count):
        total_cells = config.grid_size[0] * config.grid_size[1]
        percentage = (count.item() / total_cells) * 100
        print(f"    Sample {i}: {count.item()} / {total_cells} ({percentage:.2f}%)")
    
    # Verify correctness properties
    print("\n" + "-" * 80)
    print("Verifying Correctness Properties:")
    
    # Property 4: Output shape invariant
    H, W = config.grid_size
    expected_shape = (batch_size, out_channels, H, W)
    shape_correct = bev_features.shape == expected_shape
    print(f"  ✓ Property 4 (Output shape): {shape_correct}")
    
    # All values finite
    all_finite = torch.isfinite(bev_features).all()
    print(f"  ✓ All values finite: {all_finite}")
    
    print("\n" + "=" * 80)
    print("Demonstration complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
