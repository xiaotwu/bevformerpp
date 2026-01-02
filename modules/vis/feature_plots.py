"""
Feature map visualization utilities with shared scaling.

This module provides robust feature visualization that:
- Uses shared scaling across multiple feature maps for fair comparison
- Supports percentile-based normalization to handle outliers
- Provides multiple visualization modes (channel mean, max, PCA)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt


def compute_shared_scale(
    features: List[Optional[Union[np.ndarray, torch.Tensor]]],
    method: str = 'percentile',
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> Tuple[float, float]:
    """
    Compute shared vmin/vmax across multiple feature maps.
    
    Args:
        features: List of feature tensors (B, C, H, W) or None
        method: 'percentile' for robust scaling, 'minmax' for absolute scaling
        percentile_low: Lower percentile for clipping (default: 1%)
        percentile_high: Upper percentile for clipping (default: 99%)
    
    Returns:
        (vmin, vmax) tuple for shared scaling
    """
    all_values = []
    
    for feat in features:
        if feat is None:
            continue
        
        # Convert to numpy
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        
        # Compute channel mean
        if feat.ndim == 4:  # (B, C, H, W)
            feat = feat[0].mean(axis=0)  # (H, W)
        elif feat.ndim == 3:  # (C, H, W)
            feat = feat.mean(axis=0)  # (H, W)
        
        all_values.append(feat.flatten())
    
    if len(all_values) == 0:
        return 0.0, 1.0
    
    combined = np.concatenate(all_values)
    
    if method == 'percentile':
        vmin = np.percentile(combined, percentile_low)
        vmax = np.percentile(combined, percentile_high)
    else:  # minmax
        vmin = combined.min()
        vmax = combined.max()
    
    # Ensure valid range
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    return float(vmin), float(vmax)


def _feature_to_vis(
    feature: Optional[Union[np.ndarray, torch.Tensor]],
    mode: str = 'channel_mean',
) -> Optional[np.ndarray]:
    """
    Convert feature tensor to 2D visualization array.
    
    Args:
        feature: (B, C, H, W) or (C, H, W) tensor
        mode: 'channel_mean', 'channel_max', or 'abs_mean'
    
    Returns:
        (H, W) numpy array or None
    """
    if feature is None:
        return None
    
    if isinstance(feature, torch.Tensor):
        feature = feature.detach().cpu().numpy()
    
    # Handle batch dimension
    if feature.ndim == 4:
        feature = feature[0]  # (C, H, W)
    
    if feature.ndim != 3:
        return feature if feature.ndim == 2 else None
    
    if mode == 'channel_mean':
        return feature.mean(axis=0)
    elif mode == 'channel_max':
        return feature.max(axis=0)
    elif mode == 'abs_mean':
        return np.abs(feature).mean(axis=0)
    else:
        return feature.mean(axis=0)


def plot_bev_feature_robust(
    feature: Optional[Union[np.ndarray, torch.Tensor]],
    ax: Optional[plt.Axes] = None,
    title: str = "BEV Feature",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'viridis',
    mode: str = 'channel_mean',
    show_colorbar: bool = True,
    na_message: str = 'N/A',
) -> plt.Axes:
    """
    Plot BEV feature map with robust handling.
    
    Args:
        feature: (B, C, H, W) or (C, H, W) feature tensor
        ax: Matplotlib axes
        title: Plot title
        vmin, vmax: Value range (None = auto)
        cmap: Colormap
        mode: How to reduce channels ('channel_mean', 'channel_max', 'abs_mean')
        show_colorbar: Whether to add colorbar
        na_message: Message to show if feature is None
    
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    vis = _feature_to_vis(feature, mode)
    
    if vis is None:
        ax.text(0.5, 0.5, na_message, ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        ax.axis('off')
        return ax
    
    # Auto-compute range if not provided
    if vmin is None:
        vmin = np.percentile(vis, 1)
    if vmax is None:
        vmax = np.percentile(vis, 99)
    
    if vmax <= vmin:
        vmax = vmin + 1e-6
    
    im = ax.imshow(vis, origin='lower', cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    return ax


def plot_features_with_shared_scale(
    features: Dict[str, Optional[Union[np.ndarray, torch.Tensor]]],
    axes: Optional[List[plt.Axes]] = None,
    titles: Optional[Dict[str, str]] = None,
    cmap: str = 'viridis',
    mode: str = 'channel_mean',
    figsize: Tuple[int, int] = (15, 5),
) -> Tuple[plt.Figure, List[plt.Axes], Tuple[float, float]]:
    """
    Plot multiple feature maps with shared scaling.
    
    Args:
        features: Dict mapping names to feature tensors
        axes: Optional list of axes (created if None)
        titles: Optional dict mapping names to display titles
        cmap: Colormap
        mode: Channel reduction mode
        figsize: Figure size if creating new figure
    
    Returns:
        fig: Matplotlib figure
        axes: List of axes
        (vmin, vmax): Shared scale used
    """
    n = len(features)
    feature_list = list(features.values())
    names = list(features.keys())
    
    # Compute shared scale
    vmin, vmax = compute_shared_scale(feature_list)
    
    # Create figure if needed
    if axes is None:
        fig, axes = plt.subplots(1, n, figsize=figsize)
        if n == 1:
            axes = [axes]
    else:
        fig = axes[0].figure if len(axes) > 0 else plt.gcf()
    
    # Plot each feature
    for i, (name, feat) in enumerate(features.items()):
        title = titles.get(name, name) if titles else name
        plot_bev_feature_robust(
            feat, axes[i], title=title, vmin=vmin, vmax=vmax, 
            cmap=cmap, mode=mode, show_colorbar=(i == n-1)
        )
    
    return fig, list(axes), (vmin, vmax)


def compute_feature_metrics(
    camera_bev: Optional[torch.Tensor],
    lidar_bev: Optional[torch.Tensor],
    fused_bev: Optional[torch.Tensor],
) -> Dict:
    """
    Compute metrics comparing feature maps.
    
    Returns:
        Dict with norms and similarity metrics
    """
    metrics = {}
    
    def norm(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach()
        return float(torch.norm(x.float()).item()) if isinstance(x, torch.Tensor) else float(np.linalg.norm(x))
    
    def cosine_sim(a, b):
        if a is None or b is None:
            return None
        if isinstance(a, torch.Tensor):
            a = a.detach().flatten().float()
        else:
            a = torch.from_numpy(a.flatten()).float()
        if isinstance(b, torch.Tensor):
            b = b.detach().flatten().float()
        else:
            b = torch.from_numpy(b.flatten()).float()
        
        # Ensure same size
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        
        sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        return float(sim.item())
    
    metrics['norm_camera'] = norm(camera_bev)
    metrics['norm_lidar'] = norm(lidar_bev)
    metrics['norm_fused'] = norm(fused_bev)
    
    metrics['cos_sim_fused_camera'] = cosine_sim(fused_bev, camera_bev)
    metrics['cos_sim_fused_lidar'] = cosine_sim(fused_bev, lidar_bev)
    metrics['cos_sim_camera_lidar'] = cosine_sim(camera_bev, lidar_bev)
    
    return metrics



