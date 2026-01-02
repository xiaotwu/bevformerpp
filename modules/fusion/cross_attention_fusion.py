"""
Bidirectional Cross-Attention BEV Fusion Module.

Implements alignment-aware cross-attention fusion as specified in the project proposal:
- Camera BEV queries LiDAR BEV (semantic enrichment)
- LiDAR BEV queries Camera BEV (geometric grounding)
- Gated residual combination of both directions
- 2D sinusoidal positional encoding for spatial awareness

This is the PROPOSAL-CONSISTENT default fusion method.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for BEV features.
    
    Encodes (x, y) position using sine/cosine functions at different frequencies.
    This provides translation-equivariant spatial awareness without learnable parameters.
    """
    
    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        """
        Args:
            embed_dim: Embedding dimension (must be divisible by 4)
            temperature: Temperature for frequency scaling
        """
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sinusoidal encoding"
        self.embed_dim = embed_dim
        self.temperature = temperature
        
    def forward(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Generate 2D positional encoding.
        
        Args:
            H: Height of BEV grid
            W: Width of BEV grid
            device: Target device
            dtype: Target dtype
            
        Returns:
            Positional encoding of shape (1, embed_dim, H, W)
        """
        # Create coordinate grids
        y_coords = torch.arange(H, device=device, dtype=dtype)
        x_coords = torch.arange(W, device=device, dtype=dtype)
        
        # Normalize to [0, 1]
        y_coords = y_coords / max(H - 1, 1)
        x_coords = x_coords / max(W - 1, 1)
        
        # Create meshgrid
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Compute frequencies
        dim_per_axis = self.embed_dim // 4  # Split among sin/cos for x and y
        dim_range = torch.arange(dim_per_axis, device=device, dtype=dtype)
        freq = 1.0 / (self.temperature ** (2 * dim_range / dim_per_axis))
        
        # Compute encodings
        # Shape: (H, W, dim_per_axis)
        x_enc_sin = torch.sin(xx.unsqueeze(-1) * freq * math.pi)
        x_enc_cos = torch.cos(xx.unsqueeze(-1) * freq * math.pi)
        y_enc_sin = torch.sin(yy.unsqueeze(-1) * freq * math.pi)
        y_enc_cos = torch.cos(yy.unsqueeze(-1) * freq * math.pi)
        
        # Concatenate: (H, W, embed_dim)
        pos_enc = torch.cat([x_enc_sin, x_enc_cos, y_enc_sin, y_enc_cos], dim=-1)
        
        # Reshape to (1, embed_dim, H, W)
        pos_enc = pos_enc.permute(2, 0, 1).unsqueeze(0)
        
        return pos_enc


class CrossAttentionBlock(nn.Module):
    """
    Single cross-attention block: Query attends to Key/Value.
    
    Used bidirectionally:
    - Camera queries LiDAR (Q=cam, K/V=lidar)
    - LiDAR queries Camera (Q=lidar, K/V=cam)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_pos_encoding: bool = True
    ):
        """
        Args:
            embed_dim: Feature embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_pos_encoding: Whether to add positional encoding
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_pos_encoding = use_pos_encoding
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Positional encoding
        if use_pos_encoding:
            self.pos_encoding = SinusoidalPositionalEncoding2D(embed_dim)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query_features: torch.Tensor,
        kv_features: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Cross-attention forward pass.
        
        Args:
            query_features: Query features (B, C, H, W)
            kv_features: Key/Value features (B, C, H, W)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (attended_features, attention_weights or None)
            - attended_features: (B, C, H, W)
            - attention_weights: (B, num_heads, N, N) if return_attention else None
        """
        B, C, H, W = query_features.shape
        N = H * W
        
        # Add positional encoding if enabled
        if self.use_pos_encoding:
            pos_enc = self.pos_encoding(H, W, query_features.device, query_features.dtype)
            query_features = query_features + pos_enc
            kv_features = kv_features + pos_enc
        
        # Flatten to tokens: (B, C, H, W) -> (B, N, C)
        q_tokens = query_features.flatten(2).permute(0, 2, 1)
        kv_tokens = kv_features.flatten(2).permute(0, 2, 1)
        
        # Project
        Q = self.q_proj(q_tokens)  # (B, N, C)
        K = self.k_proj(kv_tokens)  # (B, N, C)
        V = self.v_proj(kv_tokens)  # (B, N, C)
        
        # Reshape for multi-head attention: (B, N, C) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention: (B, num_heads, N, N)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention: (B, num_heads, N, head_dim)
        attended = torch.matmul(attn_weights, V)
        
        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        attended = attended.permute(0, 2, 1, 3).reshape(B, N, C)
        
        # Output projection and residual
        attended = self.out_proj(attended)
        attended = self.dropout(attended)
        
        # Add residual and normalize
        output = self.norm(q_tokens + attended)
        
        # Reshape to spatial: (B, N, C) -> (B, C, H, W)
        output = output.permute(0, 2, 1).view(B, C, H, W)
        
        if return_attention:
            return output, attn_weights
        return output, None


class BidirectionalCrossAttentionFusion(nn.Module):
    """
    Bidirectional Cross-Attention Fusion for BEV features.
    
    Proposal-consistent implementation:
    1. Camera queries LiDAR -> semantic-enriched camera features
    2. LiDAR queries Camera -> geometry-grounded LiDAR features
    3. Gated residual combination -> fused BEV features
    
    FIX 3: Token downsampling before cross-attention prevents OOM on large BEV grids.
    With token_downsample=4, a 200x200 BEV (40k tokens) becomes 50x50 (2.5k tokens).
    
    This enables alignment-aware fusion where each modality can
    attend to complementary information in the other.
    """
    
    def __init__(
        self,
        lidar_channels: int = 64,
        camera_channels: int = 256,
        fused_channels: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_bidirectional: bool = True,
        use_gate: bool = True,
        pos_encoding: str = "sinusoidal_2d",
        token_downsample: int = 4
    ):
        """
        Args:
            lidar_channels: LiDAR BEV feature channels
            camera_channels: Camera BEV feature channels
            fused_channels: Output fused feature channels
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_bidirectional: Enable bidirectional attention (both directions)
            use_gate: Enable learned gating for residual combination
            pos_encoding: Type of positional encoding ("sinusoidal_2d" or "none")
            token_downsample: Spatial downsample factor before cross-attention (1, 2, or 4)
        """
        super().__init__()
        
        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels
        self.use_bidirectional = use_bidirectional
        self.use_gate = use_gate
        self.token_downsample = token_downsample
        
        use_pos = pos_encoding == "sinusoidal_2d"
        
        # FIX 3: Token downsampling to prevent cross-attention OOM
        if token_downsample > 1:
            self.downsample = nn.AvgPool2d(kernel_size=token_downsample, stride=token_downsample)
            # Upsample back to original resolution after attention
            self.upsample = nn.Upsample(scale_factor=token_downsample, mode='bilinear', align_corners=False)
        else:
            self.downsample = None
            self.upsample = None
        
        # Project both modalities to common dimension
        self.lidar_proj = nn.Conv2d(lidar_channels, fused_channels, kernel_size=1, bias=False)
        self.camera_proj = nn.Conv2d(camera_channels, fused_channels, kernel_size=1, bias=False)
        
        # Cross-attention: Camera queries LiDAR
        self.cam_query_lidar = CrossAttentionBlock(
            embed_dim=fused_channels,
            num_heads=num_heads,
            dropout=dropout,
            use_pos_encoding=use_pos
        )
        
        # Cross-attention: LiDAR queries Camera (if bidirectional)
        if use_bidirectional:
            self.lidar_query_cam = CrossAttentionBlock(
                embed_dim=fused_channels,
                num_heads=num_heads,
                dropout=dropout,
                use_pos_encoding=use_pos
            )
        
        # Gating network for residual combination
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Conv2d(fused_channels * 2, fused_channels, kernel_size=1),
                nn.BatchNorm2d(fused_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(fused_channels, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(
        self,
        F_lidar: torch.Tensor,
        F_cam: torch.Tensor,
        debug: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of bidirectional cross-attention fusion.
        
        Args:
            F_lidar: LiDAR BEV features (B, C1, H, W)
            F_cam: Camera BEV features (B, C2, H, W)
            debug: If True, return attention maps for visualization
            
        Returns:
            Dictionary containing:
            - 'fused': Fused BEV features (B, C3, H, W)
            - 'cam_attended': Camera features attended by LiDAR (if debug)
            - 'lidar_attended': LiDAR features attended by Camera (if debug)
            - 'attn_cam_to_lidar': Attention weights cam->lidar (if debug)
            - 'attn_lidar_to_cam': Attention weights lidar->cam (if debug)
            - 'gate': Gating weights (if debug and use_gate)
        """
        B, C1, H, W = F_lidar.shape
        _, C2, H_cam, W_cam = F_cam.shape
        
        # Verify spatial alignment
        assert H == H_cam and W == W_cam, \
            f"Spatial dimensions must match: LiDAR ({H}, {W}) vs Camera ({H_cam}, {W_cam})"
        
        # Project to common dimension
        lidar_feat = self.lidar_proj(F_lidar)  # (B, C3, H, W)
        cam_feat = self.camera_proj(F_cam)  # (B, C3, H, W)
        
        # FIX 3: Downsample tokens before cross-attention to prevent OOM
        if self.downsample is not None:
            lidar_feat_ds = self.downsample(lidar_feat)
            cam_feat_ds = self.downsample(cam_feat)
            H_ds, W_ds = lidar_feat_ds.shape[2], lidar_feat_ds.shape[3]
        else:
            lidar_feat_ds = lidar_feat
            cam_feat_ds = cam_feat
        
        result = {}
        
        # Direction 1: Camera queries LiDAR (semantic enrichment)
        # Cross-attention operates on downsampled tokens
        cam_attended, attn_c2l = self.cam_query_lidar(
            query_features=cam_feat_ds,
            kv_features=lidar_feat_ds,
            return_attention=debug
        )
        
        if self.use_bidirectional:
            # Direction 2: LiDAR queries Camera (geometric grounding)
            lidar_attended, attn_l2c = self.lidar_query_cam(
                query_features=lidar_feat_ds,
                kv_features=cam_feat_ds,
                return_attention=debug
            )
            
            # Gated residual combination
            if self.use_gate:
                # Concatenate for gating
                concat_feat = torch.cat([cam_attended, lidar_attended], dim=1)
                gate = self.gate_net(concat_feat)  # (B, 1, H_ds, W_ds)
                
                # Weighted combination: gate * cam_attended + (1-gate) * lidar_attended
                fused = gate * cam_attended + (1 - gate) * lidar_attended
                
                if debug:
                    result['gate'] = gate
            else:
                # Simple average
                fused = (cam_attended + lidar_attended) / 2
                
            if debug:
                result['lidar_attended'] = lidar_attended
                result['attn_lidar_to_cam'] = attn_l2c
        else:
            # Unidirectional: only camera queries LiDAR
            fused = cam_attended
        
        # FIX 3: Upsample back to original resolution
        if self.upsample is not None:
            fused = self.upsample(fused)
        
        # Output projection
        fused = self.output_proj(fused)
        
        result['fused'] = fused
        
        if debug:
            result['cam_attended'] = cam_attended
            result['attn_cam_to_lidar'] = attn_c2l
        
        return result


def create_cross_attention_fusion(config: dict) -> BidirectionalCrossAttentionFusion:
    """
    Factory function to create cross-attention fusion from config.
    
    Expected config keys:
        fusion.dim: int (default 256)
        fusion.num_heads: int (default 8)
        fusion.use_bidirectional: bool (default True)
        fusion.use_gate: bool (default True)
        fusion.pos_encoding: str (default "sinusoidal_2d")
        fusion.dropout: float (default 0.0)
        fusion.token_downsample: int (default 4) - FIX 3: downsample before attention
        model.lidar.num_features: int (default 64)
        model.camera.num_features: int (default 256)
    """
    fusion_cfg = config.get('fusion', {})
    model_cfg = config.get('model', {})
    
    return BidirectionalCrossAttentionFusion(
        lidar_channels=model_cfg.get('lidar', {}).get('num_features', 64),
        camera_channels=model_cfg.get('camera', {}).get('num_features', 256),
        fused_channels=fusion_cfg.get('dim', 256),
        num_heads=fusion_cfg.get('num_heads', 8),
        dropout=fusion_cfg.get('dropout', 0.0),
        use_bidirectional=fusion_cfg.get('use_bidirectional', True),
        token_downsample=fusion_cfg.get('token_downsample', 4),
        use_gate=fusion_cfg.get('use_gate', True),
        pos_encoding=fusion_cfg.get('pos_encoding', 'sinusoidal_2d')
    )

