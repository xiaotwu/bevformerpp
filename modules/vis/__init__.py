"""
Visualization utilities for BEVFormer++ project.

This module provides publication-ready visualization with:
- Strict semantic correctness (camera_only vs fusion modes)
- Proper normalization (percentile-based, shared scales)
- NMS-based peak extraction
- Clean tensor extraction interface

Main entry point:
    visualize_bevformer_case() - Single API for all visualizations

Design principles:
1. NO tensor truthiness checks (if tensor: is FORBIDDEN)
2. Explicit mode semantics (camera_only vs fusion)
3. Shared color scales for fair comparison
4. Paper-ready formatting
"""

# Core utilities
from .core import (
    # Normalization
    normalize_for_display,
    compute_bev_energy,
    compute_shared_normalization,
    # Peak extraction
    extract_nms_peaks,
    get_gt_centers,
    Peak,
    # Tensor extraction
    extract_visual_tensors,
    VisualTensors,
    # Statistics
    summarize_heatmap_stats,
    compute_tensor_stats,
)

# LiDAR utilities
from .lidar_bev import (
    project_lidar_to_bev,
    debug_lidar_bev,
    plot_lidar_bev_density,
    # Extraction helpers
    extract_lidar_points_np,
    validate_points_np,
)

# Heatmap utilities (legacy, prefer core functions)
from .heatmap_peaks import (
    extract_local_maxima,
    extract_topk_peaks_nms,
    summarize_heatmap,
    visualize_peaks_on_heatmap,
)

# Feature utilities (legacy, prefer core functions)
from .feature_plots import (
    compute_shared_scale,
    plot_features_with_shared_scale,
    plot_bev_feature_robust,
    compute_feature_metrics,
)

# Main visualization API
from .figure import (
    visualize_bevformer_case,
    create_side_by_side_comparison,
)

# Legacy paper figure (kept for backward compatibility)
from .paper_figure import (
    create_paper_figure,
    create_comparison_figure_v2,
)


__all__ = [
    # === PRIMARY API ===
    'visualize_bevformer_case',
    'create_side_by_side_comparison',
    
    # === Core utilities ===
    'normalize_for_display',
    'compute_bev_energy',
    'compute_shared_normalization',
    'compute_tensor_stats',
    'extract_nms_peaks',
    'get_gt_centers',
    'extract_visual_tensors',
    'Peak',
    'VisualTensors',
    'summarize_heatmap_stats',
    
    # === LiDAR ===
    'project_lidar_to_bev',
    'debug_lidar_bev',
    'plot_lidar_bev_density',
    'extract_lidar_points_np',
    'validate_points_np',
    
    # === Legacy (kept for backward compat) ===
    'extract_local_maxima',
    'extract_topk_peaks_nms',
    'summarize_heatmap',
    'visualize_peaks_on_heatmap',
    'compute_shared_scale',
    'plot_features_with_shared_scale',
    'plot_bev_feature_robust',
    'compute_feature_metrics',
    'create_paper_figure',
    'create_comparison_figure_v2',
]
