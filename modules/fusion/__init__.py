"""
BEV Fusion Module.

Contains fusion strategies for combining LiDAR and Camera BEV features.
"""

from .cross_attention_fusion import (
    BidirectionalCrossAttentionFusion,
    CrossAttentionBlock,
    SinusoidalPositionalEncoding2D,
    create_cross_attention_fusion,
)

# Import from the renamed impl file (was modules/fusion.py, now modules/_fusion_impl.py)
from .._fusion_impl import (
    SpatialFusionModule,
    CrossAttentionFusion,
    LocalWindowAttentionFusion,
    ConvolutionalFusion,
)

__all__ = [
    'BidirectionalCrossAttentionFusion',
    'CrossAttentionBlock',
    'SinusoidalPositionalEncoding2D',
    'create_cross_attention_fusion',
    'SpatialFusionModule',
    'CrossAttentionFusion',
    'LocalWindowAttentionFusion',
    'ConvolutionalFusion',
]

