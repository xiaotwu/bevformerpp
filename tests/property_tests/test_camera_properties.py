"""
Property-based tests for Camera BEV encoder.
Tests correctness properties defined in the design document.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra import numpy as npst

from modules.camera_encoder import (
    CameraBEVEncoder,
    project_3d_to_2d,
    backproject_2d_to_3d
)


# Hypothesis strategies for generating test data

@st.composite
def batch_sizes(draw):
    """Generate valid batch sizes."""
    return draw(st.integers(min_value=1, max_value=4))


@st.composite
def camera_images(draw, batch_size=1, n_cam=6, img_h=900, img_w=1600):
    """Generate valid camera images."""
    # Generate random images with values in [0, 1]
    images = draw(npst.arrays(
        dtype=np.float32,
        shape=(batch_size, n_cam, 3, img_h, img_w),
        elements=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    ))
    return images


@st.composite
def camera_intrinsics(draw, batch_size=1, n_cam=6):
    """Generate valid camera intrinsic matrices."""
    intrinsics = np.zeros((batch_size, n_cam, 3, 3), dtype=np.float32)
    
    for b in range(batch_size):
        for cam in range(n_cam):
            # Typical camera intrinsics for nuScenes
            fx = draw(st.floats(min_value=800.0, max_value=1200.0))
            fy = draw(st.floats(min_value=800.0, max_value=1200.0))
            cx = draw(st.floats(min_value=700.0, max_value=900.0))
            cy = draw(st.floats(min_value=400.0, max_value=500.0))
            
            intrinsics[b, cam] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
    
    return intrinsics


@st.composite
def camera_extrinsics(draw, batch_size=1, n_cam=6):
    """Generate valid camera extrinsic matrices (ego to camera)."""
    extrinsics = np.zeros((batch_size, n_cam, 4, 4), dtype=np.float32)
    
    for b in range(batch_size):
        for cam in range(n_cam):
            # Generate random rotation (simplified: only yaw)
            yaw = draw(st.floats(min_value=-np.pi, max_value=np.pi))
            
            # Generate random translation
            tx = draw(st.floats(min_value=-2.0, max_value=2.0))
            ty = draw(st.floats(min_value=-2.0, max_value=2.0))
            tz = draw(st.floats(min_value=-1.0, max_value=1.0))
            
            # Create rotation matrix (yaw only for simplicity)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            rotation = np.array([
                [cos_yaw, -sin_yaw, 0],
                [sin_yaw, cos_yaw, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            # Create 4x4 transformation matrix
            extrinsics[b, cam, :3, :3] = rotation
            extrinsics[b, cam, :3, 3] = [tx, ty, tz]
            extrinsics[b, cam, 3, 3] = 1.0
    
    return extrinsics


@st.composite
def points_3d_in_ego_frame(draw, n_points=10):
    """Generate 3D points in ego frame."""
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
        elements=st.floats(min_value=-3.0, max_value=2.0, allow_nan=False, allow_infinity=False)
    ))
    
    return np.stack([x, y, z], axis=1)


# Property 7: Camera BEV encoder output shape invariant
@given(batch_size=batch_sizes())
@settings(max_examples=100, deadline=None)
def test_property_7_camera_output_shape_invariant(batch_size):
    """
    Feature: bev-fusion-system, Property 7: Camera BEV encoder output shape invariant
    
    Validates: Requirements 2.3
    
    For any valid multi-view image input, the camera BEV encoder should output features
    with shape (B, C2, H, W) where C2, H, W match the configuration.
    """
    # Configuration
    bev_h, bev_w = 200, 200
    embed_dim = 256
    n_cam = 6
    img_h, img_w = 900, 1600
    
    # Create encoder
    encoder = CameraBEVEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        embed_dim=embed_dim,
        num_layers=2,  # Use fewer layers for faster testing
        img_h=img_h,
        img_w=img_w
    )
    encoder.eval()
    
    # Generate synthetic inputs
    images = torch.randn(batch_size, n_cam, 3, img_h, img_w)
    
    # Generate camera parameters
    intrinsics = torch.zeros(batch_size, n_cam, 3, 3)
    extrinsics = torch.zeros(batch_size, n_cam, 4, 4)
    
    for b in range(batch_size):
        for cam in range(n_cam):
            # Simple intrinsics
            intrinsics[b, cam] = torch.tensor([
                [1000.0, 0, 800.0],
                [0, 1000.0, 450.0],
                [0, 0, 1.0]
            ])
            
            # Simple extrinsics (identity with small translation)
            extrinsics[b, cam] = torch.eye(4)
            extrinsics[b, cam, 0, 3] = cam * 0.5  # Small x offset per camera
    
    # Forward pass
    with torch.no_grad():
        output = encoder(images, intrinsics, extrinsics)
    
    # Property: Output shape should match expected dimensions
    expected_shape = (batch_size, embed_dim, bev_h, bev_w)
    assert output.shape == expected_shape, \
        f"Output shape {output.shape} does not match expected {expected_shape}"
    
    # Property: All output values should be finite
    assert torch.isfinite(output).all(), \
        "All output values should be finite (no NaN or Inf)"
    
    # Property: Output should be on the correct device
    assert output.device.type in ['cpu', 'cuda'], \
        "Output should be on a valid device"
    
    # Property: Output dtype should be float
    assert output.dtype in [torch.float32, torch.float16], \
        "Output should be float type"


# Property 8: Camera projection round-trip consistency
@given(points_3d=points_3d_in_ego_frame(n_points=10))
@settings(max_examples=100, deadline=None)
def test_property_8_projection_round_trip_consistency(points_3d):
    """
    Feature: bev-fusion-system, Property 8: Camera projection round-trip consistency
    
    Validates: Requirements 2.4
    
    For any 3D point in BEV space, projecting to image space using camera parameters
    and then back-projecting to BEV should yield a point close to the original
    (within numerical tolerance).
    """
    # Create synthetic camera parameters
    intrinsics = torch.tensor([
        [1000.0, 0, 800.0],
        [0, 1000.0, 450.0],
        [0, 0, 1.0]
    ], dtype=torch.float32)
    
    # Simple extrinsic (identity)
    extrinsics = torch.eye(4, dtype=torch.float32)
    
    # Convert to torch
    points_3d_t = torch.from_numpy(points_3d)
    
    # Project to 2D
    points_2d, valid_mask = project_3d_to_2d(points_3d_t, intrinsics, extrinsics)
    
    # Filter valid points
    valid_points_3d = points_3d_t[valid_mask]
    valid_points_2d = points_2d[valid_mask]
    
    if len(valid_points_3d) == 0:
        # No valid projections, skip test
        return
    
    # Get depths for back-projection
    depths = valid_points_3d[:, 2]
    
    # Back-project to 3D
    reconstructed_3d = backproject_2d_to_3d(
        valid_points_2d,
        depths,
        intrinsics,
        extrinsics
    )
    
    # Property: Reconstructed points should match original points
    # Allow some numerical tolerance
    assert torch.allclose(reconstructed_3d, valid_points_3d, rtol=1e-3, atol=1e-3), \
        "Round-trip projection should preserve 3D coordinates within tolerance"
    
    # Property: All reconstructed values should be finite
    assert torch.isfinite(reconstructed_3d).all(), \
        "All reconstructed values should be finite"


# Additional test: Projection produces valid masks
@given(points_3d=points_3d_in_ego_frame(n_points=20))
@settings(max_examples=50, deadline=None)
def test_projection_valid_masks(points_3d):
    """
    Test that projection correctly identifies valid and invalid projections.
    """
    # Create camera parameters
    intrinsics = torch.tensor([
        [1000.0, 0, 800.0],
        [0, 1000.0, 450.0],
        [0, 0, 1.0]
    ], dtype=torch.float32)
    
    extrinsics = torch.eye(4, dtype=torch.float32)
    
    points_3d_t = torch.from_numpy(points_3d)
    
    # Project
    points_2d, valid_mask = project_3d_to_2d(points_3d_t, intrinsics, extrinsics)
    
    # Property: Valid mask should be boolean
    assert valid_mask.dtype == torch.bool, "Valid mask should be boolean"
    
    # Property: Valid mask shape should match number of points
    assert valid_mask.shape == (len(points_3d),), \
        "Valid mask shape should match number of input points"
    
    # Property: Points with negative depth should be invalid
    # Transform to camera frame to check depth
    points_3d_homo = torch.cat([
        points_3d_t,
        torch.ones(len(points_3d_t), 1)
    ], dim=-1)
    cam_coords = torch.matmul(extrinsics, points_3d_homo.T)[:3, :]
    negative_depth = cam_coords[2, :] <= 0.1
    
    # All points with negative depth should be marked invalid
    assert torch.all(~valid_mask[negative_depth]), \
        "Points with negative depth should be marked invalid"


# Additional test: BEV queries are learnable
def test_bev_queries_are_learnable():
    """
    Test that BEV queries are learnable parameters.
    """
    encoder = CameraBEVEncoder(bev_h=200, bev_w=200, embed_dim=256, num_layers=1)
    
    # Check that bev_queries is a parameter
    assert hasattr(encoder, 'bev_queries'), "Encoder should have bev_queries"
    assert isinstance(encoder.bev_queries, torch.nn.Parameter), \
        "bev_queries should be a learnable parameter"
    
    # Check shape
    assert encoder.bev_queries.shape == (200 * 200, 256), \
        "bev_queries should have shape (H*W, embed_dim)"
    
    # Check that it requires grad
    assert encoder.bev_queries.requires_grad, \
        "bev_queries should require gradients"


# Additional test: Positional encoding exists
def test_positional_encoding_exists():
    """
    Test that positional encoding is properly initialized.
    """
    encoder = CameraBEVEncoder(bev_h=200, bev_w=200, embed_dim=256, num_layers=1)
    
    # Check that bev_pos_embed exists
    assert hasattr(encoder, 'bev_pos_embed'), "Encoder should have bev_pos_embed"
    assert isinstance(encoder.bev_pos_embed, torch.nn.Parameter), \
        "bev_pos_embed should be a learnable parameter"
    
    # Check shape
    assert encoder.bev_pos_embed.shape == (200 * 200, 256), \
        "bev_pos_embed should have shape (H*W, embed_dim)"


# Additional test: Multi-layer attention
def test_multi_layer_attention():
    """
    Test that encoder has multiple attention layers as configured.
    """
    num_layers = 6
    encoder = CameraBEVEncoder(
        bev_h=200,
        bev_w=200,
        embed_dim=256,
        num_layers=num_layers
    )
    
    # Check number of attention layers
    assert len(encoder.cross_attention_layers) == num_layers, \
        f"Should have {num_layers} cross-attention layers"
    
    assert len(encoder.layer_norms) == num_layers, \
        f"Should have {num_layers} layer norms"
    
    assert len(encoder.ffns) == num_layers, \
        f"Should have {num_layers} FFN layers"
    
    assert len(encoder.ffn_norms) == num_layers, \
        f"Should have {num_layers} FFN norms"


# Additional test: BEV coordinates are properly initialized
def test_bev_coordinates_initialization():
    """
    Test that BEV grid coordinates are properly initialized.
    """
    bev_h, bev_w = 200, 200
    x_range = (-51.2, 51.2)
    y_range = (-51.2, 51.2)
    
    encoder = CameraBEVEncoder(
        bev_h=bev_h,
        bev_w=bev_w,
        bev_x_range=x_range,
        bev_y_range=y_range
    )
    
    # Check that bev_coords exists
    assert hasattr(encoder, 'bev_coords'), "Encoder should have bev_coords"
    
    # Check shape
    assert encoder.bev_coords.shape == (bev_h, bev_w, 3), \
        f"bev_coords should have shape ({bev_h}, {bev_w}, 3)"
    
    # Check value ranges
    x_coords = encoder.bev_coords[:, :, 0]
    y_coords = encoder.bev_coords[:, :, 1]
    z_coords = encoder.bev_coords[:, :, 2]
    
    assert x_coords.min() >= x_range[0] and x_coords.max() <= x_range[1], \
        "X coordinates should be within specified range"
    
    assert y_coords.min() >= y_range[0] and y_coords.max() <= y_range[1], \
        "Y coordinates should be within specified range"
    
    # Z should be constant (reference height)
    assert torch.all(z_coords == z_coords[0, 0]), \
        "Z coordinates should be constant (reference height)"
