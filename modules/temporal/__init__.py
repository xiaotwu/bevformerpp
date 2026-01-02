"""
Temporal Aggregation Modules for BEV features.

This package contains the components of the Motion-Compensated ConvRNN (MC-ConvRNN):
1. Ego-motion warping (ego_motion_warp.py)
2. Residual motion refinement (residual_motion_refine.py)
3. Visibility gating (visibility_gate.py)

The integrated MC-ConvRNN is in modules/mc_convrnn.py.
"""

from .ego_motion_warp import (
    EgoMotionWarp,
    warp_bev_with_ego_motion,
    create_bev_grid
)

from .residual_motion_refine import (
    ResidualMotionRefine,
    ResidualMotionModule,
    apply_residual_warp
)

from .visibility_gate import (
    VisibilityGate,
    VisibilityGatedUpdate,
    compute_bounds_mask,
    compute_feature_consistency_mask
)

__all__ = [
    # Ego-motion warping
    'EgoMotionWarp',
    'warp_bev_with_ego_motion',
    'create_bev_grid',
    # Residual motion
    'ResidualMotionRefine',
    'ResidualMotionModule',
    'apply_residual_warp',
    # Visibility gating
    'VisibilityGate',
    'VisibilityGatedUpdate',
    'compute_bounds_mask',
    'compute_feature_consistency_mask',
]

