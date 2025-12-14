"""
Attention modules for BEV Fusion System.
Includes spatial cross-attention, temporal self-attention, and temporal gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

class SpatialCrossAttention(nn.Module):
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
        query: (B, H*W, C) - BEV queries
        key, value: Not used directly in deformable attention (features are sampled)
        query_pos: (B, H*W, C)
        reference_points_cam: (B, H*W, num_points, 2) - Projected 2D points on image
        bev_mask: (B, H*W, num_points) - Valid mask
        """
        # Simplified implementation: Sampling from image features
        # value contains the multi-scale image features: (B * N_cam, C, H_img, W_img)
        
        B, L, C = query.shape
        N_cam = 6 # Assuming 6 cameras
        
        # Reshape value to (B, N_cam, C, H_img, W_img)
        # Note: 'value' passed here is actually the flattened features from backbone/neck
        # For simplicity in this mock-up, we'll assume 'value' is the list of feature maps
        # But standard BEVFormer passes flattened features. 
        # Let's assume 'value' is (B*N_cam, C, H_feat, W_feat) for the single scale case for now.
        
        # To make this runnable without full Deformable Attn CUDA kernel:
        # We will implement a "Grid Sample" based attention.
        
        # value: (B*N_cam, C, H_feat, W_feat)
        feat_map = value 
        B_N, C, H_feat, W_feat = feat_map.shape
        
        # reference_points_cam: (B, N_cam, H_bev, W_bev, 2) -> (u, v) normalized
        # We need to sample features at these points.
        
        # Flatten BEV dims
        H_bev = int(L**0.5) # Assuming square
        W_bev = H_bev
        
        # reference_points_cam comes in as (B, H*W, N_cam, 2) or similar? 
        # Let's align with the calling code in bevformer.py
        # In bevformer.py:
        # reference_points_cam = project_bev_to_image(...) -> (B, N_cam, H*W, 2)
        
        ref_points = reference_points_cam # (B, N_cam, L, 2)
        
        # Reshape for grid_sample: (B*N_cam, L, 1, 2)
        ref_points = rearrange(ref_points, 'b n l p -> (b n) l 1 p')
        
        # Grid sample
        # feat_map: (B*N_cam, C, H_feat, W_feat)
        sampled_feats = F.grid_sample(feat_map, ref_points, align_corners=False) # (B*N_cam, C, L, 1)
        sampled_feats = sampled_feats.squeeze(-1) # (B*N_cam, C, L)
        
        # Reshape back: (B, N_cam, C, L)
        sampled_feats = rearrange(sampled_feats, '(b n) c l -> b n l c', b=B)
        
        # Apply mask (valid projection)
        # bev_mask: (B, N_cam, L)
        mask = bev_mask.unsqueeze(-1) # (B, N_cam, L, 1)
        sampled_feats = sampled_feats * mask
        
        # Aggregate across cameras (Mean or Weighted Sum)
        # Improved: Learnable weights
        # We compute weights based on the query + sampled features
        # For simplicity, let's use a simple attention over cameras
        
        # (B, L, N_cam, C)
        sampled_feats = rearrange(sampled_feats, 'b n l c -> b l n c')
        
        # Simple mean for now, or we can add a small network to predict weights
        # output = sampled_feats.mean(dim=2) # (B, L, C)
        
        # Let's use the attention_weights layer defined in init
        # But that was for deformable attention. 
        # Let's implement a simple attention over N_cam
        
        # Query: (B, L, C) -> (B, L, 1, C)
        q = query.unsqueeze(2)
        
        # Keys: sampled_feats (B, L, N_cam, C)
        # Scores: (B, L, N_cam)
        scores = torch.sum(q * sampled_feats, dim=-1) / (C ** 0.5)
        attn = F.softmax(scores, dim=-1) # (B, L, N_cam)
        
        # Weighted sum
        output = torch.sum(attn.unsqueeze(-1) * sampled_feats, dim=2) # (B, L, C)
        
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
        self.norm = nn.LayerNorm([embed_dim])
    
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
