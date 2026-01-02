"""
Utility modules for BEV Fusion System.

Includes:
- Core geometric utilities (align_bev_features, visibility masks)
- LiDAR point/mask normalization utilities
- Visualization utilities (vis_bev)
"""

from .vis_bev import (
    BEVVisConfig,
    visualize_bev_features,
    visualize_heatmap,
    visualize_occupancy,
    visualize_boxes_bev,
    visualize_fusion_comparison,
    visualize_temporal_sequence,
    visualize_attention_map,
)

# Core geometry/utils (imported from the legacy core module)
from .core import (
    align_bev_features,
    generate_grid_from_transform,
    compute_visibility_mask,
    warp_bev,
    # LiDAR point/mask utilities
    normalize_lidar_points_and_mask,
    validate_lidar_mask_shape,
)

__all__ = [
    # Core geometry/utils
    'align_bev_features',
    'generate_grid_from_transform',
    'compute_visibility_mask',
    'warp_bev',
    # LiDAR point/mask utilities
    'normalize_lidar_points_and_mask',
    'validate_lidar_mask_shape',
    # Visualization
    'BEVVisConfig',
    'visualize_bev_features',
    'visualize_heatmap',
    'visualize_occupancy',
    'visualize_boxes_bev',
    'visualize_fusion_comparison',
    'visualize_temporal_sequence',
    'visualize_attention_map',
]

