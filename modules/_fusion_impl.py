"""
Spatial Fusion Module

This module implements spatial fusion between LiDAR and camera BEV features.

FUSION TYPES (ordered by proposal alignment):
1. "bidirectional_cross_attn" - PROPOSAL DEFAULT: Bidirectional cross-attention with gated residual
2. "cross_attention" - Legacy: Unidirectional cross-attention (LiDAR queries Camera)
3. "local_attention" - Local window attention O(N×W²) for large grids
4. "convolutional" - Conv-based fusion O(N) for resource-constrained settings

The bidirectional cross-attention fusion is the proposal-consistent default.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

# Import the new bidirectional cross-attention fusion
from .fusion.cross_attention_fusion import BidirectionalCrossAttentionFusion


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion between LiDAR and camera BEV features.

    LiDAR features attend to camera features to incorporate semantic information
    while preserving geometric structure through residual connections.

    WARNING: This implementation uses full O(N²) attention. For 200×200 BEV grids,
    this is very memory-intensive (~6GB per sample for fp32). Consider using
    SpatialFusionModule for production deployments.
    """

    def __init__(self, lidar_channels: int = 64, camera_channels: int = 256,
                 fused_channels: int = 256, num_heads: int = 8):
        """
        Args:
            lidar_channels: Number of channels in LiDAR BEV features (C1)
            camera_channels: Number of channels in camera BEV features (C2)
            fused_channels: Number of channels in fused output (C3)
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels
        self.num_heads = num_heads
        
        # Project LiDAR features to fused dimension for query
        self.lidar_proj = nn.Conv2d(lidar_channels, fused_channels, kernel_size=1, bias=False)
        
        # Project camera features to fused dimension for key and value
        self.camera_key_proj = nn.Conv2d(camera_channels, fused_channels, kernel_size=1, bias=False)
        self.camera_value_proj = nn.Conv2d(camera_channels, fused_channels, kernel_size=1, bias=False)
        
        # Multi-head attention
        assert fused_channels % num_heads == 0, "fused_channels must be divisible by num_heads"
        self.head_dim = fused_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Output projection
        self.out_proj = nn.Conv2d(fused_channels, fused_channels, kernel_size=1, bias=False)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fused_channels)
        self.norm2 = nn.LayerNorm(fused_channels)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fused_channels, fused_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fused_channels * 4, fused_channels),
            nn.Dropout(0.1)
        )
        
        # Residual projection if dimensions don't match
        if lidar_channels != fused_channels:
            self.residual_proj = nn.Conv2d(lidar_channels, fused_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(self, F_lidar: torch.Tensor, F_cam: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of cross-attention fusion.

        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features

        Returns:
            F_fused: (B, C3, H, W) - Fused BEV features
        """
        B, C1, H, W = F_lidar.shape
        _, C2, H_cam, W_cam = F_cam.shape

        # Verify spatial dimensions match
        assert H == H_cam and W == W_cam, \
            f"Spatial dimensions must match: LiDAR ({H}, {W}) vs Camera ({H_cam}, {W_cam})"

        # PERFORMANCE WARNING: O(N²) attention
        N = H * W
        if N > 10000 and not getattr(self, '_warned_once', False):
            warnings.warn(
                f"CrossAttentionFusion: Using full attention with N={N} spatial positions. "
                f"Attention matrix size: {N}×{N} = {N*N/1e9:.2f}B elements per head. "
                f"Consider using SpatialFusionModule for large grids.",
                RuntimeWarning
            )
            self._warned_once = True

        # Project features
        Q = self.lidar_proj(F_lidar)  # (B, C3, H, W)
        K = self.camera_key_proj(F_cam)  # (B, C3, H, W)
        V = self.camera_value_proj(F_cam)  # (B, C3, H, W)

        # Reshape for multi-head attention
        # (B, C3, H, W) -> (B, num_heads, H*W, head_dim)
        Q = Q.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        K = K.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        V = V.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        # Compute attention scores
        # (B, num_heads, H*W, head_dim) @ (B, num_heads, head_dim, H*W) -> (B, num_heads, H*W, H*W)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, H*W, H*W)
        
        # Apply attention to values
        # (B, num_heads, H*W, H*W) @ (B, num_heads, H*W, head_dim) -> (B, num_heads, H*W, head_dim)
        attn_out = torch.matmul(attn_weights, V)
        
        # Reshape back to spatial format
        # (B, num_heads, H*W, head_dim) -> (B, C3, H, W)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, self.fused_channels, H, W)
        
        # Output projection
        attn_out = self.out_proj(attn_out)  # (B, C3, H, W)
        
        # First residual connection with layer norm
        # Reshape for layer norm: (B, C3, H, W) -> (B, H, W, C3)
        residual = self.residual_proj(F_lidar)
        x = residual + attn_out
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C3)
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # (B, C3, H, W)
        
        # Feed-forward network with second residual connection
        # Reshape for FFN: (B, C3, H, W) -> (B, H*W, C3)
        identity = x
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, self.fused_channels)
        x = self.ffn(x)
        x = x.reshape(B, H, W, self.fused_channels).permute(0, 3, 1, 2)  # (B, C3, H, W)
        
        # Second residual connection with layer norm
        x = identity + x
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C3)
        x = self.norm2(x)
        F_fused = x.permute(0, 3, 1, 2)  # (B, C3, H, W)

        return F_fused


class LocalWindowAttentionFusion(nn.Module):
    """
    Local window attention fusion - O(N × W²) complexity where W is window size.

    Instead of full O(N²) attention, each position only attends to a local window.
    For 200×200 BEV with window_size=7, this reduces attention from 1.6B to ~2M
    elements per head (800× reduction).

    This is the RECOMMENDED alternative to CrossAttentionFusion for large grids.
    """

    def __init__(self, lidar_channels: int = 64, camera_channels: int = 256,
                 fused_channels: int = 256, num_heads: int = 8, window_size: int = 7):
        """
        Args:
            lidar_channels: Number of channels in LiDAR BEV features (C1)
            camera_channels: Number of channels in camera BEV features (C2)
            fused_channels: Number of channels in fused output (C3)
            num_heads: Number of attention heads
            window_size: Local window size (default 7 = 7×7 = 49 positions)
        """
        super().__init__()

        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.pad = window_size // 2

        assert fused_channels % num_heads == 0
        self.head_dim = fused_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections
        self.lidar_proj = nn.Conv2d(lidar_channels, fused_channels, 1, bias=False)
        self.camera_key_proj = nn.Conv2d(camera_channels, fused_channels, 1, bias=False)
        self.camera_value_proj = nn.Conv2d(camera_channels, fused_channels, 1, bias=False)
        self.out_proj = nn.Conv2d(fused_channels, fused_channels, 1, bias=False)

        # Layer norms and FFN
        self.norm1 = nn.LayerNorm(fused_channels)
        self.norm2 = nn.LayerNorm(fused_channels)
        self.ffn = nn.Sequential(
            nn.Linear(fused_channels, fused_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(fused_channels * 4, fused_channels),
            nn.Dropout(0.1)
        )

        # Residual projection
        if lidar_channels != fused_channels:
            self.residual_proj = nn.Conv2d(lidar_channels, fused_channels, 1, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, F_lidar: torch.Tensor, F_cam: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with local window attention.

        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features

        Returns:
            F_fused: (B, C3, H, W) - Fused BEV features
        """
        B, _, H, W = F_lidar.shape

        # Project features
        Q = self.lidar_proj(F_lidar)  # (B, C3, H, W)
        K = self.camera_key_proj(F_cam)  # (B, C3, H, W)
        V = self.camera_value_proj(F_cam)  # (B, C3, H, W)

        # Pad K and V for unfold
        K_padded = F.pad(K, [self.pad] * 4, mode='replicate')
        V_padded = F.pad(V, [self.pad] * 4, mode='replicate')

        # Unfold to get local windows: (B, C3, H, W) -> (B, C3*W²s, H, W)
        K_windows = F.unfold(K_padded, self.window_size, padding=0)  # (B, C3*Ws², H*W)
        V_windows = F.unfold(V_padded, self.window_size, padding=0)

        # Reshape for attention
        Ws2 = self.window_size ** 2
        K_windows = K_windows.view(B, self.fused_channels, Ws2, H * W)  # (B, C3, Ws², N)
        V_windows = V_windows.view(B, self.fused_channels, Ws2, H * W)

        # Multi-head reshape
        Q_mh = Q.view(B, self.num_heads, self.head_dim, H * W)  # (B, H, D, N)
        K_mh = K_windows.view(B, self.num_heads, self.head_dim, Ws2, H * W)  # (B, H, D, Ws², N)
        V_mh = V_windows.view(B, self.num_heads, self.head_dim, Ws2, H * W)

        # Compute attention: Q @ K^T for each position
        # Q: (B, H, D, N) -> (B, H, N, D)
        # K: (B, H, D, Ws², N) -> (B, H, N, Ws², D) -> (B, H, N, D, Ws²)
        Q_mh = Q_mh.permute(0, 1, 3, 2)  # (B, H, N, D)
        K_mh = K_mh.permute(0, 1, 4, 3, 2)  # (B, H, N, Ws², D)
        V_mh = V_mh.permute(0, 1, 4, 3, 2)  # (B, H, N, Ws², D)

        # Attention scores: (B, H, N, 1, D) @ (B, H, N, D, Ws²) -> (B, H, N, 1, Ws²)
        attn = torch.matmul(Q_mh.unsqueeze(3), K_mh.transpose(-2, -1)) * self.scale
        attn = attn.squeeze(3)  # (B, H, N, Ws²)
        attn = F.softmax(attn, dim=-1)

        # Apply attention: (B, H, N, 1, Ws²) @ (B, H, N, Ws², D) -> (B, H, N, 1, D)
        out = torch.matmul(attn.unsqueeze(3), V_mh).squeeze(3)  # (B, H, N, D)

        # Reshape back
        out = out.permute(0, 1, 3, 2).reshape(B, self.fused_channels, H, W)
        out = self.out_proj(out)

        # Residual + LayerNorm
        residual = self.residual_proj(F_lidar)
        x = residual + out
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)

        # FFN
        identity = x
        x = self.ffn(x)
        x = identity + x
        x = self.norm2(x)

        F_fused = x.permute(0, 3, 1, 2)
        return F_fused


class ConvolutionalFusion(nn.Module):
    """
    Convolutional fusion - O(N) complexity.

    Uses 1×1 convolutions and depthwise convolutions instead of attention.
    Most efficient option for very large grids or resource-constrained settings.
    Trades off some modeling capacity for speed.
    """

    def __init__(self, lidar_channels: int = 64, camera_channels: int = 256,
                 fused_channels: int = 256, kernel_size: int = 3):
        """
        Args:
            lidar_channels: Number of channels in LiDAR BEV features
            camera_channels: Number of channels in camera BEV features
            fused_channels: Number of channels in fused output
            kernel_size: Kernel size for spatial convolutions
        """
        super().__init__()

        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels

        # Project both modalities to same dimension
        self.lidar_proj = nn.Conv2d(lidar_channels, fused_channels, 1, bias=False)
        self.camera_proj = nn.Conv2d(camera_channels, fused_channels, 1, bias=False)

        # Learned attention-like gating
        self.gate_conv = nn.Sequential(
            nn.Conv2d(fused_channels * 2, fused_channels, 1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, 2, 1),  # 2 channels for lidar/camera weights
            nn.Softmax(dim=1)
        )

        # Spatial context aggregation
        padding = kernel_size // 2
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, kernel_size, padding=padding,
                      groups=fused_channels, bias=False),  # Depthwise
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, 1, bias=False),  # Pointwise
            nn.BatchNorm2d(fused_channels),
        )

        # Output refinement
        self.output_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 1),
            nn.BatchNorm2d(fused_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fused_channels, fused_channels, 1)
        )

    def forward(self, F_lidar: torch.Tensor, F_cam: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with convolutional fusion.

        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features

        Returns:
            F_fused: (B, C3, H, W) - Fused BEV features
        """
        # Project to common dimension
        lidar_feat = self.lidar_proj(F_lidar)  # (B, C3, H, W)
        camera_feat = self.camera_proj(F_cam)  # (B, C3, H, W)

        # Compute gating weights
        concat = torch.cat([lidar_feat, camera_feat], dim=1)  # (B, 2*C3, H, W)
        gates = self.gate_conv(concat)  # (B, 2, H, W)

        # Weighted combination
        fused = gates[:, 0:1] * lidar_feat + gates[:, 1:2] * camera_feat

        # Spatial context
        fused = fused + self.spatial_conv(fused)

        # Output refinement
        F_fused = self.output_conv(fused)

        return F_fused


class SpatialFusionModule(nn.Module):
    """
    Complete spatial fusion module that combines LiDAR and camera BEV features.

    This module ensures spatial alignment and applies fusion to produce unified
    BEV features that preserve both geometric and semantic information.

    Supports multiple fusion strategies via the fusion_type parameter:
    - "bidirectional_cross_attn": PROPOSAL DEFAULT - Bidirectional cross-attention with gated residual
    - "cross_attention": Legacy unidirectional cross-attention (LiDAR queries Camera)
    - "local_attention": Local window attention O(N×W²) (recommended for production)
    - "convolutional": Depthwise-separable convolutions O(N) (fastest, lowest memory)
    """

    def __init__(self, lidar_channels: int = 64, camera_channels: int = 256,
                 fused_channels: int = 256, num_heads: int = 8,
                 fusion_type: str = "bidirectional_cross_attn", window_size: int = 7,
                 use_bidirectional: bool = True, use_gate: bool = True,
                 pos_encoding: str = "sinusoidal_2d", dropout: float = 0.0,
                 token_downsample: int = 4):
        """
        Args:
            lidar_channels: Number of channels in LiDAR BEV features (C1)
            camera_channels: Number of channels in camera BEV features (C2)
            fused_channels: Number of channels in fused output (C3)
            num_heads: Number of attention heads for attention-based fusion
            fusion_type: Type of fusion to use:
                - "bidirectional_cross_attn": Bidirectional cross-attention (PROPOSAL DEFAULT)
                - "cross_attention": Legacy unidirectional cross-attention
                - "local_attention": Local window attention (good quality, moderate memory)
                - "convolutional": Conv-based fusion (fast, low memory)
            window_size: Window size for local attention (only used if fusion_type="local_attention")
            use_bidirectional: For bidirectional_cross_attn: enable both directions
            use_gate: For bidirectional_cross_attn: enable learned gating
            pos_encoding: For bidirectional_cross_attn: "sinusoidal_2d" or "none"
            dropout: Dropout rate for attention layers
            token_downsample: FIX 3 - Downsample factor before cross-attention (1, 2, or 4)
        """
        super().__init__()

        self.lidar_channels = lidar_channels
        self.camera_channels = camera_channels
        self.fused_channels = fused_channels
        self.fusion_type = fusion_type

        # Select fusion strategy
        if fusion_type == "bidirectional_cross_attn":
            # PROPOSAL-CONSISTENT DEFAULT
            self.fusion = BidirectionalCrossAttentionFusion(
                lidar_channels=lidar_channels,
                camera_channels=camera_channels,
                fused_channels=fused_channels,
                num_heads=num_heads,
                dropout=dropout,
                use_bidirectional=use_bidirectional,
                use_gate=use_gate,
                pos_encoding=pos_encoding,
                token_downsample=token_downsample
            )
        elif fusion_type == "cross_attention":
            # Legacy unidirectional
            self.fusion = CrossAttentionFusion(
                lidar_channels=lidar_channels,
                camera_channels=camera_channels,
                fused_channels=fused_channels,
                num_heads=num_heads
            )
        elif fusion_type == "local_attention":
            self.fusion = LocalWindowAttentionFusion(
                lidar_channels=lidar_channels,
                camera_channels=camera_channels,
                fused_channels=fused_channels,
                num_heads=num_heads,
                window_size=window_size
            )
        elif fusion_type == "convolutional":
            self.fusion = ConvolutionalFusion(
                lidar_channels=lidar_channels,
                camera_channels=camera_channels,
                fused_channels=fused_channels
            )
        else:
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. "
                f"Must be one of: 'bidirectional_cross_attn', 'cross_attention', 'local_attention', 'convolutional'"
            )
    
    def verify_alignment(self, F_lidar: torch.Tensor, F_cam: torch.Tensor) -> bool:
        """
        Verify that LiDAR and camera BEV features have matching spatial dimensions.
        
        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features
        
        Returns:
            bool: True if dimensions match, False otherwise
        """
        B_lidar, C_lidar, H_lidar, W_lidar = F_lidar.shape
        B_cam, C_cam, H_cam, W_cam = F_cam.shape
        
        # Check batch size
        if B_lidar != B_cam:
            return False
        
        # Check spatial dimensions
        if H_lidar != H_cam or W_lidar != W_cam:
            return False
        
        # Check channel dimensions match expected
        if C_lidar != self.lidar_channels or C_cam != self.camera_channels:
            return False
        
        return True
    
    def forward(self, F_lidar: torch.Tensor, F_cam: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """
        Forward pass of spatial fusion module.
        
        Args:
            F_lidar: (B, C1, H, W) - LiDAR BEV features
            F_cam: (B, C2, H, W) - Camera BEV features
            debug: If True and using bidirectional_cross_attn, returns dict with attention maps
        
        Returns:
            F_fused: (B, C3, H, W) - Fused BEV features
            OR dict with 'fused' key and debug info if debug=True and fusion supports it
        
        Raises:
            AssertionError: If spatial dimensions don't match
        """
        # Verify alignment
        assert self.verify_alignment(F_lidar, F_cam), \
            f"Feature dimensions don't match: LiDAR {F_lidar.shape} vs Camera {F_cam.shape}"
        
        # Apply fusion
        if self.fusion_type == "bidirectional_cross_attn":
            result = self.fusion(F_lidar, F_cam, debug=debug)
            if debug:
                return result
            return result['fused']
        else:
            F_fused = self.fusion(F_lidar, F_cam)
            return F_fused
