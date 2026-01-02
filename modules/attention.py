"""
Attention modules for BEV Fusion System.
Includes spatial cross-attention, temporal self-attention, and temporal gating.

Design Notes:
-------------
The SpatialCrossAttention implementation is a SIMPLIFIED version using PyTorch's
grid_sample instead of CUDA-optimized Deformable Attention (as in the original
BEVFormer paper). This design choice provides:

1. Portability: No custom CUDA kernels or compilation required
2. Debugging: Standard PyTorch ops are easier to debug and profile
3. Compatibility: Works on any PyTorch-supported hardware

Trade-offs vs. Original BEVFormer:
- No multi-scale deformable sampling (single scale only)
- No learned sampling offsets (fixed grid locations)
- Slightly lower accuracy but much simpler implementation

For production use with maximum performance, consider implementing:
- mmcv's MultiScaleDeformableAttention
- Custom CUDA kernels for deformable attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class SpatialCrossAttention(nn.Module):
    """
    Simplified Spatial Cross-Attention for BEV feature extraction.

    This module samples image features at projected BEV grid locations and
    aggregates them using attention-weighted fusion across cameras.

    Unlike the original BEVFormer's deformable attention:
    - Uses grid_sample for feature sampling (no learned offsets)
    - Single-scale operation (no multi-scale pyramid)
    - Softmax attention over cameras for aggregation
    """

    def __init__(self, embed_dim=256, num_heads=8, num_points=4, num_levels=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = num_levels
        
        # Learnable weights for attention
        self.attention_weights = nn.Linear(embed_dim, num_levels * num_points * num_heads)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, query_pos, reference_points_cam, bev_mask, spatial_shapes, level_start_index):
        """
        Spatial cross-attention from images to BEV.

        Args:
            query: (B, H*W, C) - BEV queries
            key: Not used (for API compatibility)
            value: (B*N_cam, C, H_feat, W_feat) - Image features
            query_pos: (B, H*W, C) - Optional positional encoding
            reference_points_cam: (B, N_cam, H*W, 2) - Normalized 2D reference points [-1, 1]
            bev_mask: (B, N_cam, H*W) - Valid projection mask
            spatial_shapes: Not used (for API compatibility)
            level_start_index: Not used (for API compatibility)

        Returns:
            output: (B, H*W, C) - BEV features sampled from images
        """
        B, L, C = query.shape

        # Infer N_cam from reference_points_cam shape instead of hard-coding
        N_cam = reference_points_cam.shape[1]

        # value: (B*N_cam, C, H_feat, W_feat)
        feat_map = value
        B_N, C_feat, H_feat, W_feat = feat_map.shape

        # Verify dimensions
        assert B_N == B * N_cam, f"Expected {B * N_cam} feature maps, got {B_N}"

        # reference_points_cam: (B, N_cam, L, 2) -> reshape for grid_sample
        ref_points = reference_points_cam  # (B, N_cam, L, 2)

        # Reshape for grid_sample: (B*N_cam, L, 1, 2)
        ref_points = rearrange(ref_points, 'b n l p -> (b n) l 1 p')

        # Grid sample features at reference points
        # feat_map: (B*N_cam, C, H_feat, W_feat)
        #
        # GRADIENT FIX: Use padding_mode='border' instead of 'zeros'.
        # With 'zeros', out-of-bounds samples return zeros with NO gradient connection.
        # With 'border', out-of-bounds samples clamp to border pixels, preserving gradient flow.
        # The attention masking will still exclude invalid samples from the weighted sum.
        sampled_feats = F.grid_sample(
            feat_map, ref_points,
            mode='bilinear', padding_mode='border', align_corners=False
        )  # (B*N_cam, C, L, 1)
        sampled_feats = sampled_feats.squeeze(-1)  # (B*N_cam, C, L)

        # Reshape back: (B, N_cam, L, C)
        sampled_feats = rearrange(sampled_feats, '(b n) c l -> b n l c', b=B, n=N_cam)

        # Normalize sampled features per BEV cell to break ray correlation.
        # Without this, all BEV cells along the same camera ray sample identical
        # image features, causing correlated outputs and ray artifacts.
        # LayerNorm forces the network to rely on query differences (which vary
        # per cell) rather than identical sampled values.
        sampled_feats = F.layer_norm(sampled_feats, [C_feat])

        # GRADIENT FIX: Do NOT zero out invalid projections here.
        # The mask multiplication (sampled_feats * mask) was breaking gradient flow because:
        # - For invalid positions (mask=0), it sets gradient to 0, blocking backprop
        # - The attention masking below already handles invalid cameras via -inf scores
        # We keep sampled_feats as-is and let attention masking do the filtering.

        # Rearrange for attention: (B, L, N_cam, C)
        sampled_feats = rearrange(sampled_feats, 'b n l c -> b l n c')

        # Compute attention weights over cameras
        # Query: (B, L, C) -> (B, L, 1, C)
        q = query.unsqueeze(2)

        # Dot-product attention scores: (B, L, N_cam)
        scores = torch.sum(q * sampled_feats, dim=-1) / (C ** 0.5)

        # Mask invalid cameras before softmax to prevent attending to invalid projections
        # bev_mask transposed: (B, L, N_cam)
        mask_transposed = bev_mask.permute(0, 2, 1)
        scores = scores.masked_fill(mask_transposed == 0, float('-inf'))

        # Softmax over cameras (handle all-masked case)
        attn = F.softmax(scores, dim=-1)  # (B, L, N_cam)

        # GRADIENT FIX: Handle all-invalid BEV positions without breaking gradient flow.
        # nan_to_num can break gradient computation. Instead, use a mask-based approach:
        # - Detect all-invalid positions (where all cameras have mask=0)
        # - For those, use uniform weights (1/N_cam) which still allows gradient flow
        #   through the sampled features (which now have border values instead of zeros)
        all_invalid = (mask_transposed.sum(dim=-1, keepdim=True) == 0)  # (B, L, 1)
        uniform_attn = torch.full_like(attn, 1.0 / N_cam)
        attn = torch.where(all_invalid.expand_as(attn), uniform_attn, attn)

        # Replace any remaining NaN values (shouldn't happen, but safety check)
        attn = torch.nan_to_num(attn, nan=1.0 / N_cam)

        # Weighted sum
        output = torch.sum(attn.unsqueeze(-1) * sampled_feats, dim=2)  # (B, L, C)
        
        return self.output_proj(output)

class TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention across temporal sequence.
    Applies attention between current BEV features and aligned history.
    
    This is a simplified version that uses standard multi-head attention.
    For efficiency, deformable attention can be used in production.
    """
    
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        """
        Initialize temporal self-attention module.
        
        Args:
            embed_dim: Embedding dimension (feature channels)
            num_heads: Number of attention heads
            dropout: Dropout rate for attention weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor, 
                query_pos: Optional[torch.Tensor] = None,
                key_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply temporal self-attention.
        
        Args:
            query: Current BEV features, shape (B, L, C) where L = H*W
            key_value: Aligned history BEV features, shape (B, T*L, C) where T is sequence length
            query_pos: Optional positional encoding for query, shape (B, L, C)
            key_pos: Optional positional encoding for key, shape (B, T*L, C)
        
        Returns:
            Attended features, shape (B, L, C)
        """
        # Add positional embeddings if provided
        q = query + query_pos if query_pos is not None else query
        k = key_value + key_pos if key_pos is not None else key_value
        v = key_value
        
        # Apply self-attention
        # query attends to key_value (history)
        attn_output, attn_weights = self.self_attn(q, k, v)
        
        # Apply dropout and residual connection
        output = query + self.dropout(attn_output)
        
        # Layer normalization
        output = self.norm(output)
        
        return output



class TemporalGating(nn.Module):
    """
    Compute confidence weights for temporal features.
    Uses feature similarity to adaptively weight past frames.
    
    The gating mechanism helps the model decide how much to trust
    each historical frame based on its similarity to the current frame.
    """
    
    def __init__(self, embed_dim: int = 256):
        """
        Initialize temporal gating module.
        
        Args:
            embed_dim: Feature dimension
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Convolutional layers to compute gating weights
        # Takes concatenated current and aligned features
        self.gate_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, kernel_size=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, current_features: torch.Tensor, 
                aligned_features: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence weights based on feature similarity.
        
        Args:
            current_features: Current BEV features, shape (B, C, H, W)
            aligned_features: Aligned historical features, shape (B, C, H, W)
        
        Returns:
            Gating weights in [0, 1], shape (B, 1, H, W)
        """
        # Concatenate current and aligned features
        concat_features = torch.cat([current_features, aligned_features], dim=1)
        
        # Compute gating weights
        gate_weights = self.gate_conv(concat_features)  # (B, 1, H, W)
        
        return gate_weights
    
    def apply_gating(self, features: torch.Tensor, 
                     gate_weights: torch.Tensor) -> torch.Tensor:
        """
        Apply gating weights to features.
        
        Args:
            features: Features to gate, shape (B, C, H, W)
            gate_weights: Gating weights, shape (B, 1, H, W)
        
        Returns:
            Gated features, shape (B, C, H, W)
        """
        return features * gate_weights



class ResidualUpdate(nn.Module):
    """
    Combine current and temporally aggregated features using residual connection.
    Allows the model to preserve current frame information while incorporating temporal context.
    """
    
    def __init__(self, embed_dim: int = 256, use_projection: bool = False):
        """
        Initialize residual update module.
        
        Args:
            embed_dim: Feature dimension
            use_projection: Whether to use a learned projection for the residual
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_projection = use_projection
        
        if use_projection:
            # Learnable projection for the temporal features
            self.projection = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim)
            )
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, current_features: torch.Tensor, 
                temporal_features: torch.Tensor,
                alpha: float = 0.5) -> torch.Tensor:
        """
        Combine current and temporal features with residual connection.
        
        Args:
            current_features: Current BEV features, shape (B, C, H, W)
            temporal_features: Temporally aggregated features, shape (B, C, H, W)
            alpha: Weighting factor for temporal features (default: 0.5)
        
        Returns:
            Combined features, shape (B, C, H, W)
        """
        # Apply projection if enabled
        if self.use_projection:
            temporal_features = self.projection(temporal_features)
        
        # Weighted combination
        output = current_features + alpha * temporal_features
        
        # Apply layer normalization (need to permute for LayerNorm)
        # LayerNorm expects (B, H, W, C)
        B, C, H, W = output.shape
        output = output.permute(0, 2, 3, 1)  # (B, H, W, C)
        output = self.norm(output)
        output = output.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        return output
