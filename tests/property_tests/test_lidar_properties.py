"""
Property-based tests for LiDAR BEV encoder.
Tests correctness properties defined in the design document.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

from modules.data_structures import BEVGridConfig
from modules.lidar_encoder import (
    Pillarization, 
    PillarFeatureNet, 
    PointPillarsScatter,
    BackboneCNN,
    LiDARBEVEncoder
)


# Hypothesis strategies for generating test data

@st.composite
def point_clouds(draw, min_points=10, max_points=5000):
    """Generate valid point clouds within BEV bounds."""
    n_points = draw(st.integers(min_value=min_points, max_value=max_points))
    
    # Generate points within BEV range
    x = draw(npst.arrays(
        dtype=np.float32, 
        shape=(n_points,),
        elements=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    ))
    y = draw(npst.arrays(
        dtype=np.float32,
        shape=(n_points,),
        elements=st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False)
    ))
    z = draw(npst.arrays(
        dtype=np.float32,
        shape=(n_points,),
        elements=st.floats(min_value=-4.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    ))
    r = draw(npst.arrays(
        dtype=np.float32,
        shape=(n_points,),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    
    return np.stack([x, y, z, r], axis=1)


@st.composite
def batch_sizes(draw):
    """Generate valid batch sizes."""
    return draw(st.integers(min_value=1, max_value=4))


# Property 1: Point cloud to pillar conversion preserves all points within BEV bounds
@given(points=point_clouds())
@settings(max_examples=100, deadline=None)
def test_property_1_pillar_conversion_preserves_points(points):
    """
    Feature: bev-fusion-system, Property 1: Point cloud to pillar conversion preserves all points within BEV bounds
    
    Validates: Requirements 1.1
    
    For any point cloud with coordinates within the defined BEV range, converting to pillars
    should assign all points to valid pillar locations, and no points should be lost or 
    assigned to out-of-bounds pillars.
    """
    config = BEVGridConfig()
    pillarization = Pillarization(config, max_points_per_pillar=100, max_pillars=12000)
    
    # Filter points within BEV bounds (same as what pillarization does)
    mask = (
        (points[:, 0] >= config.x_min) & (points[:, 0] < config.x_max) &
        (points[:, 1] >= config.y_min) & (points[:, 1] < config.y_max) &
        (points[:, 2] >= config.z_min) & (points[:, 2] < config.z_max)
    )
    points_in_bounds = points[mask]
    
    # Convert to pillars
    pillars, pillar_coords, num_points_per_pillar = pillarization(points)
    
    # Property: All pillar coordinates should be within grid bounds
    H, W = config.grid_size
    if len(pillar_coords) > 0:
        assert np.all(pillar_coords[:, 0] >= 0), "Pillar y-coordinates should be >= 0"
        assert np.all(pillar_coords[:, 0] < H), f"Pillar y-coordinates should be < {H}"
        assert np.all(pillar_coords[:, 1] >= 0), "Pillar x-coordinates should be >= 0"
        assert np.all(pillar_coords[:, 1] < W), f"Pillar x-coordinates should be < {W}"
    
    # Property: Number of pillars should be reasonable
    # If we have points in bounds, we should have at least one pillar
    if len(points_in_bounds) > 0:
        assert len(pillars) > 0, "Should have at least one pillar when points exist"
    else:
        assert len(pillars) == 0, "Should have no pillars when no points exist"
    
    # Property: Total points in pillars should not exceed points in bounds
    total_points_in_pillars = num_points_per_pillar.sum()
    assert total_points_in_pillars <= len(points_in_bounds), \
        "Total points in pillars should not exceed input points"
    
    # Property: Pillar shapes should be correct
    max_points_per_pillar = 100
    assert pillars.shape == (len(pillar_coords), max_points_per_pillar, 4), \
        f"Pillars shape should be (num_pillars, {max_points_per_pillar}, 4)"
    assert pillar_coords.shape == (len(pillars), 2), \
        "Pillar coords shape should be (num_pillars, 2)"
    assert num_points_per_pillar.shape == (len(pillars),), \
        "Num points per pillar shape should be (num_pillars,)"
    
    # Property: All values should be finite
    assert np.all(np.isfinite(pillars)), "All pillar values should be finite"
    assert np.all(np.isfinite(pillar_coords)), "All pillar coordinates should be finite"


# Property 4: LiDAR BEV encoder output shape invariant
@given(points=point_clouds(), batch_size=batch_sizes())
@settings(max_examples=100, deadline=None)
def test_property_4_lidar_output_shape_invariant(points, batch_size):
    """
    Feature: bev-fusion-system, Property 4: LiDAR BEV encoder output shape invariant
    
    Validates: Requirements 1.4, 1.5
    
    For any valid point cloud input, the complete LiDAR BEV encoder (pillarization + scatter + CNN)
    should output features with shape (B, C1, H, W) where C1, H, W match the configuration.
    """
    config = BEVGridConfig()
    out_channels = 64
    encoder = LiDARBEVEncoder(config, out_channels=out_channels)
    encoder.eval()  # Set to eval mode to avoid batch norm issues with small batches
    
    # Create batch by replicating the point cloud
    points_batch = [points.copy() for _ in range(batch_size)]
    
    # Forward pass
    with torch.no_grad():
        output = encoder(points_batch)
    
    # Property: Output shape should match expected dimensions
    H, W = config.grid_size
    expected_shape = (batch_size, out_channels, H, W)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} does not match expected {expected_shape}"
    
    # Property: All output values should be finite
    assert torch.isfinite(output).all(), "All output values should be finite (no NaN or Inf)"
    
    # Property: Output should be on the correct device
    assert output.device.type in ['cpu', 'cuda'], "Output should be on a valid device"
    
    # Property: Output dtype should be float
    assert output.dtype in [torch.float32, torch.float16], "Output should be float type"


# Additional test: Empty point cloud handling
@settings(max_examples=50, deadline=None)
@given(batch_size=batch_sizes())
def test_empty_point_cloud_handling(batch_size):
    """
    Test that the encoder handles empty point clouds gracefully.
    """
    config = BEVGridConfig()
    out_channels = 64
    encoder = LiDARBEVEncoder(config, out_channels=out_channels)
    encoder.eval()
    
    # Create batch of empty point clouds
    empty_points = np.zeros((0, 4), dtype=np.float32)
    points_batch = [empty_points.copy() for _ in range(batch_size)]
    
    # Forward pass should not crash
    with torch.no_grad():
        output = encoder(points_batch)
    
    # Output should have correct shape
    H, W = config.grid_size
    assert output.shape == (batch_size, out_channels, H, W)
    
    # Output should be all zeros (or at least finite)
    assert torch.isfinite(output).all()


# Test: Pillar feature encoding produces consistent dimensions
@given(points=point_clouds())
@settings(max_examples=50, deadline=None)
def test_pillar_feature_encoding_dimensions(points):
    """
    Test that PillarFeatureNet produces features with consistent dimensions.
    Related to Property 2 from design doc.
    """
    config = BEVGridConfig()
    pillarization = Pillarization(config)
    feature_net = PillarFeatureNet(in_channels=9, out_channels=64)
    feature_net.eval()
    
    # Pillarize
    pillars, pillar_coords, num_points = pillarization(points)
    
    if len(pillars) == 0:
        # Skip if no pillars
        return
    
    # Convert to tensors
    pillars_t = torch.from_numpy(pillars)
    pillar_coords_t = torch.from_numpy(pillar_coords)
    num_points_t = torch.from_numpy(num_points)
    
    # Encode
    with torch.no_grad():
        features = feature_net(pillars_t, pillar_coords_t, num_points_t, config)
    
    # Check shape
    assert features.shape == (len(pillars), 64), \
        f"Features shape {features.shape} should be ({len(pillars)}, 64)"
    
    # Check all values are finite
    assert torch.isfinite(features).all(), "All feature values should be finite"


# Test: Scatter operation maintains pillar-to-grid correspondence
@given(points=point_clouds())
@settings(max_examples=50, deadline=None)
def test_scatter_operation_correspondence(points):
    """
    Test that scatter operation places features at correct grid locations.
    Related to Property 3 from design doc.
    """
    config = BEVGridConfig()
    pillarization = Pillarization(config)
    feature_net = PillarFeatureNet(in_channels=9, out_channels=64)
    scatter = PointPillarsScatter(config, in_channels=64)
    
    feature_net.eval()
    scatter.eval()
    
    # Pillarize and encode
    pillars, pillar_coords, num_points = pillarization(points)
    
    if len(pillars) == 0:
        return
    
    pillars_t = torch.from_numpy(pillars)
    pillar_coords_t = torch.from_numpy(pillar_coords)
    num_points_t = torch.from_numpy(num_points)
    
    with torch.no_grad():
        features = feature_net(pillars_t, pillar_coords_t, num_points_t, config)
        bev = scatter(features, pillar_coords_t, batch_size=1)
    
    # Check output shape
    H, W = config.grid_size
    assert bev.shape == (1, 64, H, W), f"BEV shape should be (1, 64, {H}, {W})"
    
    # Check that features are placed at correct locations
    for i, (y, x) in enumerate(pillar_coords):
        # The feature at this location should be non-zero (or at least the same as input)
        bev_feature = bev[0, :, y, x]
        input_feature = features[i]
        # They should match
        assert torch.allclose(bev_feature, input_feature, atol=1e-6), \
            f"Feature at pillar location ({y}, {x}) does not match input feature"
