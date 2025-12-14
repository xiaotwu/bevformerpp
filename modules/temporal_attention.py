"""
Temporal Aggregation Module using Transformer-based Attention.
Integrates MemoryBank, ego-motion alignment, temporal self-attention, 
temporal gating, and residual updates.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from einops import rearrange

from .memory_bank import MemoryBank
from .attention import TemporalSelfAttention, TemporalGating, ResidualUpdate
from .utils import align_bev_features


class TemporalAggregationModule(nn.Module):
    """
    Complete temporal aggregation module using transformer-based attention.
    
    This module:
    1. Stores past BEV features in a memory bank
    2. Aligns past features to current frame using ego-motion
    3. Applies temporal self-attention across the sequence
    4. Computes confidence weights via temporal gating
    5. Combines current and temporal features with residual connection
    
    Implements Requirements 4.1-4.5 from the design document.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        max_history: int = 5,
        dropout: float = 0.1,
        use_gating: bool = True,
        use_projection: bool = False,
        temporal_alpha: float = 0.5
    ):
        """
        Initialize temporal aggregation module.
        
        Args:
            embed_dim: Feature dimension (C)
            num_heads: Number of attention heads
            max_history: Maximum number of past frames to store
            dropout: Dropout rate for attention
            use_gating: Whether to use temporal gating
            use_projection: Whether to use learned projection in residual update
            temporal_alpha: Weighting factor for temporal features
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_history = max_history
        self.use_gating = use_gating
        self.temporal_alpha = temporal_alpha
        
        # Memory bank for storing past features
        self.memory_bank = MemoryBank(max_length=max_history)
        
        # Temporal self-attention
        self.temporal_attention = TemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Temporal gating (optional)
        if use_gating:
            self.temporal_gating = TemporalGating(embed_dim=embed_dim)
        
        # Residual update
        self.residual_update = ResidualUpdate(
            embed_dim=embed_dim,
            use_projection=use_projection
        )
        
        # Positional encoding for BEV queries (learnable)
        self.bev_pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1))
    
    def forward(
        self,
        current_bev: torch.Tensor,
        ego_transform: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Apply temporal aggregation to current BEV features.
        
        Args:
            current_bev: Current BEV features, shape (B, C, H, W)
            ego_transform: Ego-motion transform from previous to current frame,
                          shape (B, 4, 4). If None, no alignment is performed.
            return_attention: Whether to return attention weights (for visualization)
        
        Returns:
            Temporally enhanced BEV features, shape (B, C, H, W)
        """
        B, C, H, W = current_bev.shape
        device = current_bev.device
        
        # If memory bank is empty, just return current features
        if self.memory_bank.is_empty():
            # Store current features for next iteration
            self.memory_bank.push(current_bev, ego_transform)
            return current_bev
        
        # Get history from memory bank
        history_features, history_transforms = self.memory_bank.get_sequence()
        
        # Align historical features to current frame
        aligned_history = []
        for i, hist_feat in enumerate(history_features[:-1]):  # Exclude current frame
            if ego_transform is not None and i < len(history_transforms):
                # Align using ego-motion
                aligned_feat = align_bev_features(hist_feat, history_transforms[i])
            else:
                # No alignment if transform not available
                aligned_feat = hist_feat
            aligned_history.append(aligned_feat)
        
        # If no history available, return current features
        if len(aligned_history) == 0:
            self.memory_bank.push(current_bev, ego_transform)
            return current_bev
        
        # Stack aligned history: (T-1, B, C, H, W)
        aligned_history = torch.stack(aligned_history, dim=0)
        T_hist = aligned_history.shape[0]
        
        # Reshape for attention: (B, T-1, C, H, W) -> (B, T-1*H*W, C)
        aligned_history_flat = rearrange(aligned_history, 't b c h w -> b (t h w) c')
        
        # Reshape current BEV: (B, C, H, W) -> (B, H*W, C)
        current_bev_flat = rearrange(current_bev, 'b c h w -> b (h w) c')
        
        # Apply temporal self-attention
        # Current frame attends to aligned history
        attended_features = self.temporal_attention(
            query=current_bev_flat,
            key_value=aligned_history_flat,
            query_pos=None,  # Can add positional encoding if needed
            key_pos=None
        )
        
        # Reshape back to spatial: (B, H*W, C) -> (B, C, H, W)
        attended_features = rearrange(attended_features, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Apply temporal gating if enabled
        if self.use_gating:
            # Compute gating weights based on similarity
            gate_weights = self.temporal_gating(current_bev, attended_features)
            
            # Apply gating
            attended_features = self.temporal_gating.apply_gating(
                attended_features, 
                gate_weights
            )
        
        # Combine with current features using residual connection
        output = self.residual_update(
            current_features=current_bev,
            temporal_features=attended_features,
            alpha=self.temporal_alpha
        )
        
        # Update memory bank with current features
        self.memory_bank.push(current_bev, ego_transform)
        
        return output
    
    def reset(self):
        """Reset the memory bank (e.g., at scene boundaries)."""
        self.memory_bank.clear()
    
    def get_memory_length(self) -> int:
        """Get the current number of stored frames."""
        return len(self.memory_bank)


class TemporalAggregationWrapper(nn.Module):
    """
    Wrapper for temporal aggregation that handles batch processing.
    Useful for training and evaluation loops.
    """
    
    def __init__(self, temporal_module: TemporalAggregationModule):
        """
        Initialize wrapper.
        
        Args:
            temporal_module: The temporal aggregation module to wrap
        """
        super().__init__()
        self.temporal_module = temporal_module
    
    def forward(
        self,
        bev_features: torch.Tensor,
        ego_transforms: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Process a batch of BEV features with temporal aggregation.
        
        Args:
            bev_features: BEV features, shape (B, C, H, W)
            ego_transforms: Optional ego-motion transforms, shape (B, 4, 4)
        
        Returns:
            Temporally enhanced features, shape (B, C, H, W)
        """
        return self.temporal_module(bev_features, ego_transforms)
    
    def reset_all(self):
        """Reset memory for all sequences."""
        self.temporal_module.reset()
