"""
Tests for MC-ConvRNN and its components.

Tests the three proposal-mandated mechanisms:
1. Ego-motion warping
2. Residual motion refinement
3. Visibility gating
"""

import pytest
import torch
import math

from modules.temporal.ego_motion_warp import (
    EgoMotionWarp,
    warp_bev_with_ego_motion,
    create_bev_grid
)
from modules.temporal.residual_motion_refine import (
    ResidualMotionRefine,
    ResidualMotionModule,
    apply_residual_warp
)
from modules.temporal.visibility_gate import (
    VisibilityGate,
    VisibilityGatedUpdate,
    compute_bounds_mask,
    compute_feature_consistency_mask
)
from modules.mc_convrnn import MCConvRNN, ConvGRUCell


class TestEgoMotionWarp:
    """Tests for ego-motion warping module."""
    
    def test_identity_transform_unchanged(self):
        """Test that identity transform leaves features unchanged."""
        B, C, H, W = 2, 64, 50, 50
        
        features = torch.randn(B, C, H, W)
        identity = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        warped = warp_bev_with_ego_motion(features, identity)
        
        # Should be very close to original
        assert torch.allclose(warped, features, atol=1e-5), \
            "Identity transform should not change features"
    
    def test_output_shape(self):
        """Test that output shape matches input shape."""
        B, C, H, W = 2, 64, 100, 100
        
        features = torch.randn(B, C, H, W)
        # Small translation
        transform = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
        transform[:, 0, 3] = 1.0  # 1 meter x translation
        
        warped = warp_bev_with_ego_motion(features, transform)
        
        assert warped.shape == (B, C, H, W)
    
    def test_translation_shifts_features(self):
        """Test that translation actually shifts features."""
        B, C, H, W = 1, 1, 100, 100
        
        # Create a feature with a known pattern
        features = torch.zeros(B, C, H, W)
        features[0, 0, H//2, W//2] = 1.0  # Single peak at center
        
        # Translate by 10 meters in x direction (should shift ~20 pixels at 0.5m resolution)
        transform = torch.eye(4).unsqueeze(0)
        transform[0, 0, 3] = 5.0  # 5 meters
        
        warped = warp_bev_with_ego_motion(features, transform, bev_range=(-25.6, 25.6, -25.6, 25.6))
        
        # Peak should have moved
        assert warped[0, 0, H//2, W//2] < 0.5, "Peak should have shifted"
    
    def test_module_interface(self):
        """Test EgoMotionWarp module interface."""
        B, C, H, W = 2, 64, 50, 50
        
        warp_module = EgoMotionWarp(bev_range=(-51.2, 51.2, -51.2, 51.2))
        features = torch.randn(B, C, H, W)
        transform = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        warped = warp_module(features, transform)
        
        assert warped.shape == (B, C, H, W)


class TestResidualMotionRefine:
    """Tests for residual motion refinement module."""
    
    def test_flow_output_shape(self):
        """Test that flow estimator produces correct shape."""
        B, C, H, W = 2, 256, 50, 50
        
        refine = ResidualMotionRefine(in_channels=C, hidden_channels=128)
        current = torch.randn(B, C, H, W)
        warped = torch.randn(B, C, H, W)
        
        flow = refine(current, warped)
        
        assert flow.shape == (B, 2, H, W), f"Expected (B, 2, H, W), got {flow.shape}"
    
    def test_flow_bounded(self):
        """Test that flow values are bounded by max_offset."""
        B, C, H, W = 2, 256, 50, 50
        max_offset = 0.1
        
        refine = ResidualMotionRefine(in_channels=C, max_offset=max_offset)
        current = torch.randn(B, C, H, W) * 10  # Large input
        warped = torch.randn(B, C, H, W) * 10
        
        flow = refine(current, warped)
        
        assert flow.abs().max() <= max_offset + 1e-5, \
            f"Flow should be bounded by {max_offset}, got max {flow.abs().max()}"
    
    def test_zero_initialization(self):
        """Test that flow starts near zero."""
        B, C, H, W = 2, 256, 50, 50
        
        refine = ResidualMotionRefine(in_channels=C)
        
        # Check output layer is zero-initialized
        assert torch.allclose(refine.conv3.weight, torch.zeros_like(refine.conv3.weight))
        assert torch.allclose(refine.conv3.bias, torch.zeros_like(refine.conv3.bias))
    
    def test_residual_module_integration(self):
        """Test complete ResidualMotionModule."""
        B, C, H, W = 2, 256, 50, 50
        
        module = ResidualMotionModule(in_channels=C)
        current = torch.randn(B, C, H, W)
        warped = torch.randn(B, C, H, W)
        
        refined = module(current, warped)
        
        assert refined.shape == (B, C, H, W)
    
    def test_residual_module_with_flow(self):
        """Test ResidualMotionModule returns flow when requested."""
        B, C, H, W = 2, 256, 50, 50
        
        module = ResidualMotionModule(in_channels=C)
        current = torch.randn(B, C, H, W)
        warped = torch.randn(B, C, H, W)
        
        refined, flow = module(current, warped, return_flow=True)
        
        assert refined.shape == (B, C, H, W)
        assert flow.shape == (B, 2, H, W)


class TestVisibilityGate:
    """Tests for visibility gating module."""
    
    def test_bounds_mask_identity(self):
        """Test bounds mask with identity transform."""
        B, H, W = 2, 50, 50
        
        identity = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        mask = compute_bounds_mask(identity, H, W)
        
        # Identity should give all-ones mask
        assert mask.shape == (B, 1, H, W)
        assert (mask == 1.0).all(), "Identity transform should give all-ones mask"
    
    def test_bounds_mask_large_translation(self):
        """Test bounds mask with large translation."""
        B, H, W = 2, 50, 50
        
        # Large translation that moves grid out of bounds
        transform = torch.eye(4).unsqueeze(0).expand(B, -1, -1).clone()
        transform[:, 0, 3] = 100.0  # 100 meters (out of 51.2m range)
        
        mask = compute_bounds_mask(transform, H, W)
        
        # Should have significant zeros
        assert mask.sum() < B * H * W, "Large translation should create zero regions"
    
    def test_consistency_mask_identical_features(self):
        """Test consistency mask with identical features."""
        B, C, H, W = 2, 64, 50, 50
        
        features = torch.randn(B, C, H, W)
        mask = compute_feature_consistency_mask(features, features)
        
        # Identical features should give high consistency
        assert mask.shape == (B, 1, H, W)
        assert mask.mean() > 0.8, "Identical features should have high consistency"
    
    def test_consistency_mask_different_features(self):
        """Test consistency mask with very different features."""
        B, C, H, W = 2, 64, 50, 50
        
        current = torch.randn(B, C, H, W)
        warped = -current  # Opposite features
        
        mask = compute_feature_consistency_mask(current, warped)
        
        # Should have low consistency
        assert mask.mean() < 0.5, "Opposite features should have low consistency"
    
    def test_visibility_gate_modes(self):
        """Test all visibility gate modes."""
        B, C, H, W = 2, 256, 50, 50
        
        current = torch.randn(B, C, H, W)
        warped = torch.randn(B, C, H, W)
        transform = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        for mode in ["bounds", "consistency", "learned", "combined"]:
            gate = VisibilityGate(mode=mode, in_channels=C)
            mask = gate(current, warped, transform)
            
            assert mask.shape == (B, 1, H, W), f"Mode {mode} failed shape check"
            assert (mask >= 0).all() and (mask <= 1).all(), \
                f"Mode {mode} mask should be in [0, 1]"
    
    def test_visibility_gated_update(self):
        """Test visibility-gated RNN update."""
        B, C, H, W = 2, 128, 50, 50
        
        update_module = VisibilityGatedUpdate(hidden_channels=C)
        
        new_state = torch.randn(B, C, H, W)
        prev_state = torch.randn(B, C, H, W)
        current = torch.randn(B, C, H, W)
        warped = torch.randn(B, C, H, W)
        transform = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        output = update_module(new_state, prev_state, current, warped, transform)
        
        assert output.shape == (B, C, H, W)


class TestMCConvRNN:
    """Tests for integrated MC-ConvRNN module."""
    
    def test_first_frame_output(self):
        """Test MC-ConvRNN with first frame (no previous features)."""
        B, C, H, W = 2, 256, 50, 50
        
        mc_convrnn = MCConvRNN(input_channels=C, hidden_channels=128)
        current = torch.randn(B, C, H, W)
        
        output, hidden = mc_convrnn(current)
        
        assert output.shape == (B, C, H, W)
        assert hidden.shape == (B, 128, H, W)
    
    def test_sequential_frames(self):
        """Test MC-ConvRNN with sequential frames."""
        B, C, H, W = 2, 256, 50, 50
        
        mc_convrnn = MCConvRNN(input_channels=C, hidden_channels=128)
        
        # First frame
        frame1 = torch.randn(B, C, H, W)
        out1, hidden1 = mc_convrnn(frame1)
        
        # Second frame with ego-motion
        frame2 = torch.randn(B, C, H, W)
        ego_motion = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        out2, hidden2 = mc_convrnn(frame2, frame1, hidden1, ego_motion)
        
        assert out2.shape == (B, C, H, W)
        assert hidden2.shape == (B, 128, H, W)
    
    def test_ablation_disable_warping(self):
        """Test MC-ConvRNN with warping disabled."""
        B, C, H, W = 2, 256, 30, 30
        
        mc_convrnn = MCConvRNN(
            input_channels=C,
            hidden_channels=128,
            disable_warping=True
        )
        
        current = torch.randn(B, C, H, W)
        prev = torch.randn(B, C, H, W)
        hidden = torch.randn(B, 128, H, W)
        ego_motion = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        output, _ = mc_convrnn(current, prev, hidden, ego_motion)
        
        assert output.shape == (B, C, H, W)
    
    def test_ablation_disable_motion_field(self):
        """Test MC-ConvRNN with motion field disabled."""
        B, C, H, W = 2, 256, 30, 30
        
        mc_convrnn = MCConvRNN(
            input_channels=C,
            hidden_channels=128,
            disable_motion_field=True
        )
        
        current = torch.randn(B, C, H, W)
        prev = torch.randn(B, C, H, W)
        hidden = torch.randn(B, 128, H, W)
        ego_motion = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        output, _ = mc_convrnn(current, prev, hidden, ego_motion)
        
        assert output.shape == (B, C, H, W)
    
    def test_ablation_disable_visibility(self):
        """Test MC-ConvRNN with visibility gating disabled."""
        B, C, H, W = 2, 256, 30, 30
        
        mc_convrnn = MCConvRNN(
            input_channels=C,
            hidden_channels=128,
            disable_visibility=True
        )
        
        current = torch.randn(B, C, H, W)
        prev = torch.randn(B, C, H, W)
        hidden = torch.randn(B, 128, H, W)
        ego_motion = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        output, _ = mc_convrnn(current, prev, hidden, ego_motion)
        
        assert output.shape == (B, C, H, W)
    
    def test_gradient_flow(self):
        """Test that gradients flow through MC-ConvRNN."""
        B, C, H, W = 2, 256, 20, 20
        
        mc_convrnn = MCConvRNN(input_channels=C, hidden_channels=64)
        
        current = torch.randn(B, C, H, W, requires_grad=True)
        prev = torch.randn(B, C, H, W, requires_grad=True)
        hidden = torch.randn(B, 64, H, W, requires_grad=True)
        ego_motion = torch.eye(4).unsqueeze(0).expand(B, -1, -1)
        
        output, _ = mc_convrnn(current, prev, hidden, ego_motion)
        loss = output.sum()
        loss.backward()
        
        assert current.grad is not None
        assert prev.grad is not None


class TestConvGRUCell:
    """Tests for ConvGRU cell."""
    
    def test_output_shape(self):
        """Test ConvGRU output shape."""
        B, C_in, C_h, H, W = 2, 256, 128, 50, 50
        
        cell = ConvGRUCell(input_channels=C_in, hidden_channels=C_h)
        x = torch.randn(B, C_in, H, W)
        h = torch.randn(B, C_h, H, W)
        
        h_new = cell(x, h)
        
        assert h_new.shape == (B, C_h, H, W)
    
    def test_hidden_init(self):
        """Test hidden state initialization."""
        B, H, W = 2, 50, 50
        C_h = 128
        
        cell = ConvGRUCell(input_channels=256, hidden_channels=C_h)
        h_init = cell.init_hidden(B, H, W, torch.device('cpu'))
        
        assert h_init.shape == (B, C_h, H, W)
        assert (h_init == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

