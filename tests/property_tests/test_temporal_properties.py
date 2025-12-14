"""
Property-based tests for temporal aggregation modules.
Tests correctness properties for temporal attention, alignment, and gating.
"""

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st
import hypothesis.extra.numpy as npst

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from modules.utils import align_bev_features, compute_visibility_mask
from modules.attention import TemporalGating, ResidualUpdate
from modules.temporal_attention import TemporalAggregationModule
from modules.memory_bank import MemoryBank


# Hypothesis strategies for generating test data

@st.composite
def bev_features(draw, batch_size=None, channels=None, height=None, width=None):
    """Generate random BEV features."""
    B = draw(st.integers(min_value=1, max_value=4)) if batch_size is None else batch_size
    C = draw(st.integers(min_value=32, max_value=256)) if channels is None else channels
    H = draw(st.integers(min_value=50, max_value=200)) if height is None else height
    W = draw(st.integers(min_value=50, max_value=200)) if width is None else width
    
    # Generate features with reasonable values
    features = draw(npst.arrays(
        dtype=np.float32,
        shape=(B, C, H, W),
        elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
    ))
    
    return torch.from_numpy(features)


@st.composite
def ego_motion_transform(draw, batch_size=1):
    """Generate random SE(3) ego-motion transformation."""
    B = batch_size
    
    # Generate random translation (small movements)
    tx = draw(st.floats(min_value=-5.0, max_value=5.0))
    ty = draw(st.floats(min_value=-5.0, max_value=5.0))
    tz = draw(st.floats(min_value=-0.5, max_value=0.5))
    
    # Generate random yaw rotation (around z-axis)
    yaw = draw(st.floats(min_value=-np.pi, max_value=np.pi))
    
    # Create SE(3) transformation matrix
    transforms = []
    for _ in range(B):
        T = np.eye(4, dtype=np.float32)
        
        # Rotation matrix (yaw around z-axis)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        T[0, 0] = cos_yaw
        T[0, 1] = -sin_yaw
        T[1, 0] = sin_yaw
        T[1, 1] = cos_yaw
        
        # Translation
        T[0, 3] = tx
        T[1, 3] = ty
        T[2, 3] = tz
        
        transforms.append(T)
    
    return torch.from_numpy(np.stack(transforms, axis=0))


# Property Tests

@given(
    features=bev_features(batch_size=2, channels=64, height=100, width=100),
    ego_transform=ego_motion_transform(batch_size=2)
)
@settings(max_examples=100, deadline=None)
def test_property_11_temporal_alignment_preserves_dimensions(features, ego_transform):
    """
    Feature: bev-fusion-system, Property 11: Temporal alignment preserves feature dimensions
    
    For any sequence of BEV features and corresponding ego-motion transforms,
    aligning all features to the current frame should produce aligned features
    with the same shape as the original features.
    
    Validates: Requirements 4.1
    """
    # Get input shape
    B, C, H, W = features.shape
    
    # Apply alignment
    aligned_features = align_bev_features(features, ego_transform)
    
    # Property 1: Output shape must match input shape
    assert aligned_features.shape == features.shape, \
        f"Shape mismatch: expected {features.shape}, got {aligned_features.shape}"
    
    # Property 2: All values must be finite (no NaN or Inf)
    assert torch.isfinite(aligned_features).all(), \
        "Aligned features contain NaN or Inf values"
    
    # Property 3: Batch dimension preserved
    assert aligned_features.shape[0] == B, \
        f"Batch size changed: expected {B}, got {aligned_features.shape[0]}"
    
    # Property 4: Channel dimension preserved
    assert aligned_features.shape[1] == C, \
        f"Channel dimension changed: expected {C}, got {aligned_features.shape[1]}"
    
    # Property 5: Spatial dimensions preserved
    assert aligned_features.shape[2] == H and aligned_features.shape[3] == W, \
        f"Spatial dimensions changed: expected ({H}, {W}), got ({aligned_features.shape[2]}, {aligned_features.shape[3]})"


@given(
    features=bev_features(batch_size=2, channels=64, height=100, width=100)
)
@settings(max_examples=100, deadline=None)
def test_property_11_identity_transform_preserves_features(features):
    """
    Test that identity transformation preserves features (special case of Property 11).
    
    When ego-motion is identity (no movement), aligned features should be
    very close to original features.
    """
    B = features.shape[0]
    
    # Create identity transformation
    identity_transform = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)
    
    # Apply alignment with identity transform
    aligned_features = align_bev_features(features, identity_transform)
    
    # Features should be very close to original (allowing for numerical errors)
    # Note: grid_sample may introduce small interpolation errors
    assert torch.allclose(aligned_features, features, rtol=1e-3, atol=1e-3), \
        "Identity transform should preserve features (within numerical tolerance)"


@given(
    current_features=bev_features(batch_size=2, channels=64, height=100, width=100),
    aligned_features=bev_features(batch_size=2, channels=64, height=100, width=100)
)
@settings(max_examples=100, deadline=None)
def test_property_13_temporal_gating_weights_bounded(current_features, aligned_features):
    """
    Feature: bev-fusion-system, Property 13: Temporal gating weights are bounded
    
    For any temporal attention outputs, the computed gating weights should be
    in the range [0, 1] for all spatial locations.
    
    Validates: Requirements 4.3
    """
    # Create temporal gating module
    C = current_features.shape[1]
    gating_module = TemporalGating(embed_dim=C)
    gating_module.eval()
    
    # Compute gating weights
    with torch.no_grad():
        gate_weights = gating_module(current_features, aligned_features)
    
    # Property 1: Output shape should be (B, 1, H, W)
    B, _, H, W = current_features.shape
    assert gate_weights.shape == (B, 1, H, W), \
        f"Gate weights shape mismatch: expected ({B}, 1, {H}, {W}), got {gate_weights.shape}"
    
    # Property 2: All weights must be in [0, 1]
    assert (gate_weights >= 0.0).all(), \
        f"Gate weights contain values < 0: min = {gate_weights.min().item()}"
    assert (gate_weights <= 1.0).all(), \
        f"Gate weights contain values > 1: max = {gate_weights.max().item()}"
    
    # Property 3: All values must be finite
    assert torch.isfinite(gate_weights).all(), \
        "Gate weights contain NaN or Inf values"


@given(
    features=bev_features(batch_size=2, channels=64, height=100, width=100)
)
@settings(max_examples=50, deadline=None)
def test_property_13_gating_extreme_cases(features):
    """
    Test gating with extreme cases (identical and very different features).
    """
    C = features.shape[1]
    gating_module = TemporalGating(embed_dim=C)
    gating_module.eval()
    
    with torch.no_grad():
        # Case 1: Identical features (should have high confidence)
        gate_weights_identical = gating_module(features, features)
        assert (gate_weights_identical >= 0.0).all() and (gate_weights_identical <= 1.0).all()
        
        # Case 2: Very different features (zeros vs features)
        zeros = torch.zeros_like(features)
        gate_weights_different = gating_module(features, zeros)
        assert (gate_weights_different >= 0.0).all() and (gate_weights_different <= 1.0).all()


@given(
    ego_transform=ego_motion_transform(batch_size=2)
)
@settings(max_examples=100, deadline=None)
def test_visibility_mask_properties(ego_transform):
    """
    Test that visibility masks have valid properties.
    """
    B = ego_transform.shape[0]
    H, W = 100, 100
    
    # Compute visibility mask
    visibility_mask = compute_visibility_mask(ego_transform, H, W)
    
    # Property 1: Shape should be (B, 1, H, W)
    assert visibility_mask.shape == (B, 1, H, W), \
        f"Visibility mask shape mismatch: expected ({B}, 1, {H}, {W}), got {visibility_mask.shape}"
    
    # Property 2: All values in [0, 1]
    assert (visibility_mask >= 0.0).all() and (visibility_mask <= 1.0).all(), \
        "Visibility mask values must be in [0, 1]"
    
    # Property 3: All values are finite
    assert torch.isfinite(visibility_mask).all(), \
        "Visibility mask contains NaN or Inf"


@given(
    current_features=bev_features(batch_size=2, channels=64, height=100, width=100),
    temporal_features=bev_features(batch_size=2, channels=64, height=100, width=100)
)
@settings(max_examples=100, deadline=None)
def test_residual_update_preserves_dimensions(current_features, temporal_features):
    """
    Test that residual update preserves feature dimensions.
    Related to Property 14 from design document.
    """
    C = current_features.shape[1]
    residual_module = ResidualUpdate(embed_dim=C)
    residual_module.eval()
    
    with torch.no_grad():
        output = residual_module(current_features, temporal_features)
    
    # Property 1: Output shape matches input shape
    assert output.shape == current_features.shape, \
        f"Output shape mismatch: expected {current_features.shape}, got {output.shape}"
    
    # Property 2: All values are finite
    assert torch.isfinite(output).all(), \
        "Output contains NaN or Inf values"


@given(
    features=bev_features(batch_size=1, channels=64, height=100, width=100)
)
@settings(max_examples=50, deadline=None)
def test_memory_bank_fifo_behavior(features):
    """
    Test that MemoryBank implements proper FIFO behavior.
    """
    max_length = 3
    memory_bank = MemoryBank(max_length=max_length)
    
    # Push features multiple times
    for i in range(max_length + 2):
        feat = features + i  # Make each feature unique
        memory_bank.push(feat)
    
    # Should only store max_length features
    assert len(memory_bank) == max_length, \
        f"MemoryBank should store at most {max_length} features, got {len(memory_bank)}"
    
    # Get sequence
    stored_features, _ = memory_bank.get_sequence()
    assert len(stored_features) == max_length, \
        f"Stored features length should be {max_length}, got {len(stored_features)}"


@given(
    features=bev_features(batch_size=1, channels=64, height=50, width=50),
    ego_transform=ego_motion_transform(batch_size=1)
)
@settings(max_examples=30, deadline=None)
def test_temporal_aggregation_module_integration(features, ego_transform):
    """
    Integration test for the complete temporal aggregation module.
    Tests that all components work together correctly.
    """
    C = features.shape[1]
    
    # Create temporal aggregation module
    temporal_module = TemporalAggregationModule(
        embed_dim=C,
        num_heads=4,
        max_history=3,
        use_gating=True
    )
    temporal_module.eval()
    
    with torch.no_grad():
        # First frame (no history)
        output1 = temporal_module(features, ego_transform)
        assert output1.shape == features.shape
        assert torch.isfinite(output1).all()
        
        # Second frame (with history)
        features2 = features + 0.1
        output2 = temporal_module(features2, ego_transform)
        assert output2.shape == features2.shape
        assert torch.isfinite(output2).all()
        
        # Third frame
        features3 = features + 0.2
        output3 = temporal_module(features3, ego_transform)
        assert output3.shape == features3.shape
        assert torch.isfinite(output3).all()
    
    # Check memory bank state
    assert len(temporal_module.memory_bank) > 0, \
        "Memory bank should contain features after processing"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
