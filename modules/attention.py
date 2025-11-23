import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

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
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, query, history_bev, query_pos, history_pos):
        """
        query: (B, L, C)
        history_bev: (B, L, C)
        """
        # In standard BEVFormer, this aligns history to current using ego-motion
        # and then performs attention.
        # Here we assume history_bev is already aligned (warped).
        
        # Concatenate query and history for key/value? 
        # Or just use history as Key/Value?
        # Standard BEVFormer uses history as KV, query as Q.
        
        # Add positional embeddings
        q = query + query_pos if query_pos is not None else query
        k = history_bev + history_pos if history_pos is not None else history_bev
        v = history_bev
        
        output, _ = self.self_attn(q, k, v)
        return output
