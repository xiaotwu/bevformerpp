"""
Tests for Bidirectional Cross-Attention Fusion Module.

Tests shape correctness, forward pass stability, and configuration options.
"""

import pytest
import torch

from modules.fusion.cross_attention_fusion import (
    BidirectionalCrossAttentionFusion,
    CrossAttentionBlock,
    SinusoidalPositionalEncoding2D,
    create_cross_attention_fusion
)
from modules.fusion import SpatialFusionModule


class TestSinusoidalPositionalEncoding2D:
    """Tests for 2D sinusoidal positional encoding."""
    
    def test_output_shape(self):
        """Test that positional encoding has correct shape."""
        embed_dim = 256
        H, W = 50, 50
        
        pos_enc = SinusoidalPositionalEncoding2D(embed_dim)
        output = pos_enc(H, W, torch.device('cpu'), torch.float32)
        
        assert output.shape == (1, embed_dim, H, W), f"Expected (1, {embed_dim}, {H}, {W}), got {output.shape}"
    
    def test_different_sizes(self):
        """Test positional encoding with different grid sizes."""
        embed_dim = 128
        pos_enc = SinusoidalPositionalEncoding2D(embed_dim)
        
        for H, W in [(100, 100), (50, 75), (200, 200)]:
            output = pos_enc(H, W, torch.device('cpu'), torch.float32)
            assert output.shape == (1, embed_dim, H, W)
    
    def test_no_nan(self):
        """Test that positional encoding has no NaN values."""
        embed_dim = 256
        pos_enc = SinusoidalPositionalEncoding2D(embed_dim)
        output = pos_enc(100, 100, torch.device('cpu'), torch.float32)
        
        assert not torch.isnan(output).any(), "Positional encoding contains NaN"


class TestCrossAttentionBlock:
    """Tests for single cross-attention block."""
    
    def test_output_shape(self):
        """Test that cross-attention block preserves shape."""
        B, C, H, W = 2, 256, 50, 50
        
        block = CrossAttentionBlock(embed_dim=C, num_heads=8)
        query = torch.randn(B, C, H, W)
        kv = torch.randn(B, C, H, W)
        
        output, _ = block(query, kv)
        
        assert output.shape == (B, C, H, W), f"Expected ({B}, {C}, {H}, {W}), got {output.shape}"
    
    def test_attention_weights_shape(self):
        """Test attention weights shape when debug enabled."""
        B, C, H, W = 2, 256, 20, 20
        num_heads = 8
        N = H * W
        
        block = CrossAttentionBlock(embed_dim=C, num_heads=num_heads)
        query = torch.randn(B, C, H, W)
        kv = torch.randn(B, C, H, W)
        
        _, attn_weights = block(query, kv, return_attention=True)
        
        expected_shape = (B, num_heads, N, N)
        assert attn_weights.shape == expected_shape, f"Expected {expected_shape}, got {attn_weights.shape}"
    
    def test_no_pos_encoding(self):
        """Test block without positional encoding."""
        B, C, H, W = 2, 256, 30, 30
        
        block = CrossAttentionBlock(embed_dim=C, num_heads=8, use_pos_encoding=False)
        query = torch.randn(B, C, H, W)
        kv = torch.randn(B, C, H, W)
        
        output, _ = block(query, kv)
        
        assert output.shape == (B, C, H, W)


class TestBidirectionalCrossAttentionFusion:
    """Tests for bidirectional cross-attention fusion."""
    
    def test_output_shape(self):
        """Test that fusion produces correct output shape."""
        B = 2
        lidar_ch, cam_ch, fused_ch = 64, 256, 256
        H, W = 50, 50
        
        fusion = BidirectionalCrossAttentionFusion(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch,
            num_heads=8
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W)
        F_cam = torch.randn(B, cam_ch, H, W)
        
        result = fusion(F_lidar, F_cam)
        
        assert 'fused' in result, "Result must contain 'fused' key"
        assert result['fused'].shape == (B, fused_ch, H, W), \
            f"Expected ({B}, {fused_ch}, {H}, {W}), got {result['fused'].shape}"
    
    def test_debug_output(self):
        """Test debug output contains expected keys."""
        B, lidar_ch, cam_ch, fused_ch = 2, 64, 256, 256
        H, W = 30, 30
        
        fusion = BidirectionalCrossAttentionFusion(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch,
            use_bidirectional=True,
            use_gate=True
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W)
        F_cam = torch.randn(B, cam_ch, H, W)
        
        result = fusion(F_lidar, F_cam, debug=True)
        
        expected_keys = {'fused', 'cam_attended', 'lidar_attended', 'gate', 
                         'attn_cam_to_lidar', 'attn_lidar_to_cam'}
        assert expected_keys.issubset(set(result.keys())), \
            f"Missing keys: {expected_keys - set(result.keys())}"
    
    def test_unidirectional_mode(self):
        """Test unidirectional mode (camera queries LiDAR only)."""
        B, lidar_ch, cam_ch, fused_ch = 2, 64, 256, 256
        H, W = 30, 30
        
        fusion = BidirectionalCrossAttentionFusion(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch,
            use_bidirectional=False
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W)
        F_cam = torch.randn(B, cam_ch, H, W)
        
        result = fusion(F_lidar, F_cam)
        
        assert result['fused'].shape == (B, fused_ch, H, W)
    
    def test_no_gate(self):
        """Test fusion without gating."""
        B, lidar_ch, cam_ch, fused_ch = 2, 64, 256, 256
        H, W = 30, 30
        
        fusion = BidirectionalCrossAttentionFusion(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch,
            use_bidirectional=True,
            use_gate=False
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W)
        F_cam = torch.randn(B, cam_ch, H, W)
        
        result = fusion(F_lidar, F_cam)
        
        assert result['fused'].shape == (B, fused_ch, H, W)
    
    def test_gate_values_valid(self):
        """Test that gate values are in [0, 1] range."""
        B, lidar_ch, cam_ch, fused_ch = 2, 64, 256, 256
        H, W = 30, 30
        
        fusion = BidirectionalCrossAttentionFusion(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch,
            use_gate=True
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W)
        F_cam = torch.randn(B, cam_ch, H, W)
        
        result = fusion(F_lidar, F_cam, debug=True)
        gate = result['gate']
        
        assert (gate >= 0).all() and (gate <= 1).all(), "Gate values must be in [0, 1]"
    
    def test_no_nan_in_output(self):
        """Test that output contains no NaN values."""
        B, lidar_ch, cam_ch, fused_ch = 2, 64, 256, 256
        H, W = 30, 30
        
        fusion = BidirectionalCrossAttentionFusion(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W)
        F_cam = torch.randn(B, cam_ch, H, W)
        
        result = fusion(F_lidar, F_cam)
        
        assert not torch.isnan(result['fused']).any(), "Output contains NaN values"
    
    def test_gradient_flow(self):
        """Test that gradients flow through the fusion."""
        B, lidar_ch, cam_ch, fused_ch = 2, 64, 256, 256
        H, W = 20, 20
        
        fusion = BidirectionalCrossAttentionFusion(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W, requires_grad=True)
        F_cam = torch.randn(B, cam_ch, H, W, requires_grad=True)
        
        result = fusion(F_lidar, F_cam)
        loss = result['fused'].sum()
        loss.backward()
        
        assert F_lidar.grad is not None, "No gradient for LiDAR input"
        assert F_cam.grad is not None, "No gradient for Camera input"


class TestSpatialFusionModuleIntegration:
    """Integration tests for SpatialFusionModule with bidirectional cross-attention."""
    
    def test_default_is_bidirectional(self):
        """Test that default fusion type is bidirectional_cross_attn."""
        fusion = SpatialFusionModule(
            lidar_channels=64,
            camera_channels=256,
            fused_channels=256
        )
        
        assert fusion.fusion_type == "bidirectional_cross_attn", \
            f"Default should be 'bidirectional_cross_attn', got '{fusion.fusion_type}'"
    
    def test_forward_pass(self):
        """Test forward pass through SpatialFusionModule."""
        B = 2
        lidar_ch, cam_ch, fused_ch = 64, 256, 256
        H, W = 50, 50
        
        fusion = SpatialFusionModule(
            lidar_channels=lidar_ch,
            camera_channels=cam_ch,
            fused_channels=fused_ch,
            fusion_type="bidirectional_cross_attn"
        )
        
        F_lidar = torch.randn(B, lidar_ch, H, W)
        F_cam = torch.randn(B, cam_ch, H, W)
        
        output = fusion(F_lidar, F_cam)
        
        assert output.shape == (B, fused_ch, H, W)


class TestCreateCrossAttentionFusion:
    """Tests for factory function."""
    
    def test_from_config(self):
        """Test creating fusion from config dict."""
        config = {
            'fusion': {
                'dim': 256,
                'num_heads': 8,
                'use_bidirectional': True,
                'use_gate': True,
                'pos_encoding': 'sinusoidal_2d',
                'dropout': 0.1
            },
            'model': {
                'lidar': {'num_features': 64},
                'camera': {'num_features': 256}
            }
        }
        
        fusion = create_cross_attention_fusion(config)
        
        assert isinstance(fusion, BidirectionalCrossAttentionFusion)
        assert fusion.fused_channels == 256
        assert fusion.use_bidirectional is True
        assert fusion.use_gate is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

