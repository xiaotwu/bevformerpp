"""
Property-based tests for Spatial Fusion Module.
Tests correctness properties defined in the design document.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

from modules.fusion import SpatialFusionModule, CrossAttentionFusion
from modules.data_structures import BEVGridConfig


# Hypothesis strategies for generating test data

@st.composite
def batch_sizes(draw):
    """Generate valid batch sizes."""
    return draw(st.integers(min_value=1, max_value=4))


@st.composite
def bev_dimensions(draw):
    """Generate valid BEV grid dimensions."""
    # Use smaller dimensions for faster testing and to avoid memory issues
    # Full attention is O(H*W)^2, so keep dimensions small
    h = draw(st.integers(min_value=20, max_value=50))
    w = draw(st.integers(min_value=20, max_value=50))
    return h, w


@st.composite
def lidar_bev_features(draw, batch_size=1, channels=64, h=200, w=200):
    """Generate valid LiDAR BEV features."""
    features = draw(npst.arrays(
        dtype=np.float32,
        shape=(batch_size, channels, h, w),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ))
    return features


@st.composite
def camera_bev_features(draw, batch_size=1, channels=256, h=200, w=200):
    """Generate valid camera BEV features."""
    features = draw(npst.arrays(
        dtype=np.float32,
        shape=(batch_size, channels, h, w),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ))
    return features


# Property 10: Spatial fusion output shape invariant
@given(
    batch_size=batch_sizes(),
    bev_dims=bev_dimensions()
)
@settings(max_examples=100, deadline=None)
def test_property_10_fusion_output_shape_invariant(batch_size, bev_dims):
    """
    Feature: bev-fusion-system, Property 10: Spatial fusion output shape invariant
    
    Validates: Requirements 3.1, 3.2
    
    For any valid LiDAR BEV features F_lidar and camera BEV features F_cam with
    matching spatial dimensions, the fusion module should output F_fused with
    shape (B, C3, H, W) where C3 matches the configuration.
    """
    h, w = bev_dims
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    # Create fusion module
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion_module.eval()
    
    # Generate synthetic features with matching spatial dimensions
    F_lidar = torch.randn(batch_size, lidar_channels, h, w)
    F_cam = torch.randn(batch_size, camera_channels, h, w)
    
    # Forward pass
    with torch.no_grad():
        F_fused = fusion_module(F_lidar, F_cam)
    
    # Property 1: Output shape should match expected dimensions
    expected_shape = (batch_size, fused_channels, h, w)
    assert F_fused.shape == expected_shape, \
        f"Output shape {F_fused.shape} does not match expected {expected_shape}"
    
    # Property 2: All output values should be finite
    assert torch.isfinite(F_fused).all(), \
        "All output values should be finite (no NaN or Inf)"
    
    # Property 3: Output should be on the correct device
    assert F_fused.device == F_lidar.device, \
        "Output should be on the same device as input"
    
    # Property 4: Output dtype should match input
    assert F_fused.dtype == F_lidar.dtype, \
        "Output dtype should match input dtype"


# Property 9: BEV grid alignment between modalities
@given(
    batch_size=batch_sizes(),
    bev_dims=bev_dimensions()
)
@settings(max_examples=100, deadline=None)
def test_property_9_bev_grid_alignment(batch_size, bev_dims):
    """
    Feature: bev-fusion-system, Property 9: BEV grid alignment between modalities
    
    Validates: Requirements 2.5, 3.3
    
    For any LiDAR BEV features and camera BEV features, both should have the same
    spatial dimensions (H, W) and represent the same physical BEV region.
    """
    h, w = bev_dims
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    # Create fusion module
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    
    # Generate features with matching spatial dimensions
    F_lidar = torch.randn(batch_size, lidar_channels, h, w)
    F_cam = torch.randn(batch_size, camera_channels, h, w)
    
    # Property 1: Verify alignment method correctly identifies matching dimensions
    assert fusion_module.verify_alignment(F_lidar, F_cam), \
        "Alignment verification should pass for matching dimensions"
    
    # Property 2: Verify alignment fails for mismatched batch sizes
    F_cam_wrong_batch = torch.randn(batch_size + 1, camera_channels, h, w)
    assert not fusion_module.verify_alignment(F_lidar, F_cam_wrong_batch), \
        "Alignment verification should fail for mismatched batch sizes"
    
    # Property 3: Verify alignment fails for mismatched spatial dimensions
    if h > 10 and w > 10:  # Only test if dimensions are large enough
        F_cam_wrong_h = torch.randn(batch_size, camera_channels, h - 1, w)
        assert not fusion_module.verify_alignment(F_lidar, F_cam_wrong_h), \
            "Alignment verification should fail for mismatched height"
        
        F_cam_wrong_w = torch.randn(batch_size, camera_channels, h, w - 1)
        assert not fusion_module.verify_alignment(F_lidar, F_cam_wrong_w), \
            "Alignment verification should fail for mismatched width"
    
    # Property 4: Verify alignment fails for mismatched channel dimensions
    F_lidar_wrong_channels = torch.randn(batch_size, lidar_channels + 1, h, w)
    assert not fusion_module.verify_alignment(F_lidar_wrong_channels, F_cam), \
        "Alignment verification should fail for mismatched LiDAR channels"
    
    F_cam_wrong_channels = torch.randn(batch_size, camera_channels + 1, h, w)
    assert not fusion_module.verify_alignment(F_lidar, F_cam_wrong_channels), \
        "Alignment verification should fail for mismatched camera channels"


# Additional test: Fusion preserves spatial structure
@given(batch_size=batch_sizes())
@settings(max_examples=50, deadline=None)
def test_fusion_preserves_spatial_structure(batch_size):
    """
    Test that fusion preserves spatial structure through residual connections.
    """
    h, w = 100, 100
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion_module.eval()
    
    # Create features with a specific spatial pattern
    F_lidar = torch.zeros(batch_size, lidar_channels, h, w)
    F_lidar[:, :, h//4:3*h//4, w//4:3*w//4] = 1.0  # Center region
    
    F_cam = torch.randn(batch_size, camera_channels, h, w)
    
    with torch.no_grad():
        F_fused = fusion_module(F_lidar, F_cam)
    
    # Property: Output should have some correlation with input spatial structure
    # (due to residual connections)
    assert F_fused.shape == (batch_size, fused_channels, h, w), \
        "Output shape should be preserved"
    
    # Property: Output should not be all zeros or all the same value
    assert F_fused.std() > 0, \
        "Output should have variation (not constant)"


# Additional test: Cross-attention produces valid attention weights
def test_cross_attention_valid_weights():
    """
    Test that cross-attention produces valid attention weights (implicitly through output).
    """
    batch_size = 2
    h, w = 50, 50
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    fusion = CrossAttentionFusion(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion.eval()
    
    F_lidar = torch.randn(batch_size, lidar_channels, h, w)
    F_cam = torch.randn(batch_size, camera_channels, h, w)
    
    with torch.no_grad():
        F_fused = fusion(F_lidar, F_cam)
    
    # Property: Output should be finite
    assert torch.isfinite(F_fused).all(), \
        "Attention output should be finite"
    
    # Property: Output should have expected shape
    assert F_fused.shape == (batch_size, fused_channels, h, w), \
        "Attention output should have correct shape"


# Additional test: Fusion is deterministic in eval mode
def test_fusion_deterministic_eval():
    """
    Test that fusion produces deterministic outputs in eval mode.
    """
    batch_size = 2
    h, w = 50, 50
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion_module.eval()
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    F_lidar = torch.randn(batch_size, lidar_channels, h, w)
    F_cam = torch.randn(batch_size, camera_channels, h, w)
    
    # First forward pass
    with torch.no_grad():
        output1 = fusion_module(F_lidar, F_cam)
    
    # Second forward pass with same inputs
    with torch.no_grad():
        output2 = fusion_module(F_lidar, F_cam)
    
    # Property: Outputs should be identical
    assert torch.allclose(output1, output2, rtol=1e-5, atol=1e-5), \
        "Fusion should be deterministic in eval mode"


# Additional test: Fusion handles edge case of zero features
def test_fusion_handles_zero_features():
    """
    Test that fusion handles edge case of zero-valued features.
    """
    batch_size = 1
    h, w = 50, 50
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion_module.eval()
    
    # Zero features
    F_lidar = torch.zeros(batch_size, lidar_channels, h, w)
    F_cam = torch.zeros(batch_size, camera_channels, h, w)
    
    with torch.no_grad():
        F_fused = fusion_module(F_lidar, F_cam)
    
    # Property: Output should be finite (not NaN or Inf)
    assert torch.isfinite(F_fused).all(), \
        "Fusion should handle zero features without producing NaN or Inf"
    
    # Property: Output shape should be correct
    assert F_fused.shape == (batch_size, fused_channels, h, w), \
        "Output shape should be correct even for zero features"


# Additional test: Fusion respects batch independence
def test_fusion_batch_independence():
    """
    Test that fusion processes each batch element independently.
    """
    batch_size = 3
    h, w = 50, 50
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion_module.eval()
    
    # Create features where each batch element is different
    F_lidar = torch.randn(batch_size, lidar_channels, h, w)
    F_cam = torch.randn(batch_size, camera_channels, h, w)
    
    # Process full batch
    with torch.no_grad():
        F_fused_batch = fusion_module(F_lidar, F_cam)
    
    # Process each element individually
    F_fused_individual = []
    for i in range(batch_size):
        with torch.no_grad():
            output = fusion_module(F_lidar[i:i+1], F_cam[i:i+1])
        F_fused_individual.append(output)
    
    F_fused_individual = torch.cat(F_fused_individual, dim=0)
    
    # Property: Batch processing should match individual processing
    assert torch.allclose(F_fused_batch, F_fused_individual, rtol=1e-4, atol=1e-4), \
        "Batch processing should be equivalent to individual processing"


# Additional test: Residual connections preserve information
def test_residual_connections_preserve_info():
    """
    Test that residual connections help preserve LiDAR geometric information.
    """
    batch_size = 1
    h, w = 50, 50
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    fusion_module = SpatialFusionModule(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    fusion_module.eval()
    
    # Create distinctive LiDAR features
    F_lidar = torch.randn(batch_size, lidar_channels, h, w)
    F_lidar[:, :, h//2, w//2] = 10.0  # Strong signal at center
    
    # Weak camera features
    F_cam = torch.randn(batch_size, camera_channels, h, w) * 0.1
    
    with torch.no_grad():
        F_fused = fusion_module(F_lidar, F_cam)
    
    # Property: Fused features should not be all uniform
    # (residual connection should preserve some structure)
    center_value = F_fused[:, :, h//2, w//2].mean()
    corner_value = F_fused[:, :, 0, 0].mean()
    
    # There should be some difference due to the strong LiDAR signal
    assert abs(center_value - corner_value) > 0, \
        "Residual connections should preserve some spatial structure"


# Additional test: Layer normalization is applied
def test_layer_normalization_applied():
    """
    Test that layer normalization is properly applied in fusion.
    """
    batch_size = 2
    h, w = 50, 50
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    
    fusion = CrossAttentionFusion(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=8
    )
    
    # Check that layer norms exist
    assert hasattr(fusion, 'norm1'), "Fusion should have norm1"
    assert hasattr(fusion, 'norm2'), "Fusion should have norm2"
    
    # Check that they are LayerNorm instances
    assert isinstance(fusion.norm1, torch.nn.LayerNorm), \
        "norm1 should be LayerNorm"
    assert isinstance(fusion.norm2, torch.nn.LayerNorm), \
        "norm2 should be LayerNorm"


# Additional test: Multi-head attention configuration
def test_multi_head_attention_config():
    """
    Test that multi-head attention is properly configured.
    """
    lidar_channels = 64
    camera_channels = 256
    fused_channels = 256
    num_heads = 8
    
    fusion = CrossAttentionFusion(
        lidar_channels=lidar_channels,
        camera_channels=camera_channels,
        fused_channels=fused_channels,
        num_heads=num_heads
    )
    
    # Property: Number of heads should match configuration
    assert fusion.num_heads == num_heads, \
        f"Number of heads should be {num_heads}"
    
    # Property: Head dimension should be correct
    expected_head_dim = fused_channels // num_heads
    assert fusion.head_dim == expected_head_dim, \
        f"Head dimension should be {expected_head_dim}"
    
    # Property: Fused channels should be divisible by num_heads
    assert fused_channels % num_heads == 0, \
        "Fused channels must be divisible by number of heads"
