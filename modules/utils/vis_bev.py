"""
BEV Visualization Utilities.

Provides visualization functions for:
- BEV features (camera, LiDAR, fused)
- Heatmaps (classification targets/predictions)
- Occupancy grids
- Ground truth and predicted 3D boxes
- Alignment verification

Outputs are saved to outputs/vis/ by default.
"""

import os
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


@dataclass
class BEVVisConfig:
    """Configuration for BEV visualization."""
    output_dir: str = "outputs/vis"
    fig_size: Tuple[int, int] = (10, 10)
    dpi: int = 150
    cmap_features: str = "viridis"
    cmap_heatmap: str = "hot"
    cmap_occupancy: str = "gray"
    box_linewidth: float = 2.0
    box_gt_color: str = "green"
    box_pred_color: str = "red"
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)


def ensure_output_dir(path: str):
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array, handling device and grad."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Normalize features to [0, 1] for visualization."""
    fmin, fmax = features.min(), features.max()
    if fmax - fmin > 1e-8:
        return (features - fmin) / (fmax - fmin)
    return features * 0


def visualize_bev_features(
    features: torch.Tensor,
    title: str = "BEV Features",
    channel_idx: Optional[int] = None,
    save_path: Optional[str] = None,
    config: Optional[BEVVisConfig] = None
) -> plt.Figure:
    """
    Visualize BEV feature map.
    
    Args:
        features: BEV features (B, C, H, W) or (C, H, W) or (H, W)
        title: Plot title
        channel_idx: If provided, visualize single channel. Otherwise, average channels.
        save_path: If provided, save figure to this path
        config: Visualization config
        
    Returns:
        matplotlib Figure
    """
    config = config or BEVVisConfig()
    features = tensor_to_numpy(features)
    
    # Handle different input shapes
    if features.ndim == 4:
        features = features[0]  # Take first batch
    if features.ndim == 3:
        if channel_idx is not None:
            features = features[channel_idx]
        else:
            features = features.mean(axis=0)  # Average over channels
    
    features = normalize_features(features)
    
    fig, ax = plt.subplots(figsize=config.fig_size, dpi=config.dpi)
    
    # Plot with BEV range as extent
    x_min, x_max, y_min, y_max = config.bev_range
    im = ax.imshow(
        features,
        cmap=config.cmap_features,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower'
    )
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Activation')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark ego vehicle position
    ax.plot(0, 0, 'r*', markersize=15, label='Ego')
    ax.legend(loc='upper right')
    
    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or config.output_dir)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def visualize_heatmap(
    heatmap: torch.Tensor,
    class_idx: Optional[int] = None,
    title: str = "Classification Heatmap",
    save_path: Optional[str] = None,
    config: Optional[BEVVisConfig] = None
) -> plt.Figure:
    """
    Visualize classification heatmap.
    
    Args:
        heatmap: Heatmap tensor (B, K, H, W) or (K, H, W) or (H, W)
        class_idx: If provided, show specific class. Otherwise, show max across classes.
        title: Plot title
        save_path: Save path
        config: Visualization config
        
    Returns:
        matplotlib Figure
    """
    config = config or BEVVisConfig()
    heatmap = tensor_to_numpy(heatmap)
    
    # Handle different input shapes
    if heatmap.ndim == 4:
        heatmap = heatmap[0]
    if heatmap.ndim == 3:
        if class_idx is not None:
            heatmap = heatmap[class_idx]
        else:
            heatmap = heatmap.max(axis=0)  # Max over classes
    
    fig, ax = plt.subplots(figsize=config.fig_size, dpi=config.dpi)
    
    x_min, x_max, y_min, y_max = config.bev_range
    im = ax.imshow(
        heatmap,
        cmap=config.cmap_heatmap,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        vmin=0,
        vmax=1
    )
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Confidence')
    
    ax.plot(0, 0, 'c*', markersize=15, label='Ego')
    ax.legend(loc='upper right')
    
    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or config.output_dir)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def visualize_occupancy(
    occupancy: torch.Tensor,
    title: str = "Occupancy Grid",
    save_path: Optional[str] = None,
    config: Optional[BEVVisConfig] = None
) -> plt.Figure:
    """
    Visualize occupancy grid from LiDAR.
    
    Args:
        occupancy: Occupancy tensor (B, 1, H, W) or (H, W)
        title: Plot title
        save_path: Save path
        config: Visualization config
        
    Returns:
        matplotlib Figure
    """
    config = config or BEVVisConfig()
    occupancy = tensor_to_numpy(occupancy)
    
    # Handle different shapes
    while occupancy.ndim > 2:
        occupancy = occupancy[0]
    
    fig, ax = plt.subplots(figsize=config.fig_size, dpi=config.dpi)
    
    x_min, x_max, y_min, y_max = config.bev_range
    im = ax.imshow(
        occupancy,
        cmap=config.cmap_occupancy,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower'
    )
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Occupancy')
    
    ax.plot(0, 0, 'r*', markersize=15, label='Ego')
    ax.legend(loc='upper right')
    
    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or config.output_dir)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def visualize_boxes_bev(
    gt_boxes: Optional[List[Dict]] = None,
    pred_boxes: Optional[List[Dict]] = None,
    background: Optional[torch.Tensor] = None,
    title: str = "BEV Detections",
    save_path: Optional[str] = None,
    config: Optional[BEVVisConfig] = None
) -> plt.Figure:
    """
    Visualize ground truth and predicted boxes in BEV.
    
    Args:
        gt_boxes: List of GT box dicts with 'x', 'y', 'w', 'l', 'yaw' keys
        pred_boxes: List of predicted box dicts
        background: Optional background feature map
        title: Plot title
        save_path: Save path
        config: Visualization config
        
    Returns:
        matplotlib Figure
    """
    config = config or BEVVisConfig()
    
    fig, ax = plt.subplots(figsize=config.fig_size, dpi=config.dpi)
    
    x_min, x_max, y_min, y_max = config.bev_range
    
    # Draw background if provided
    if background is not None:
        bg = tensor_to_numpy(background)
        while bg.ndim > 2:
            bg = bg.mean(axis=0) if bg.ndim == 3 else bg[0]
        bg = normalize_features(bg)
        ax.imshow(
            bg,
            cmap='gray',
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',
            alpha=0.5
        )
    
    def draw_rotated_box(ax, x, y, w, l, yaw, color, label=None):
        """Draw a rotated rectangle."""
        # Create rectangle corners
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Half dimensions
        hw, hl = w / 2, l / 2
        
        # Corners in local frame
        corners_local = np.array([
            [-hl, -hw],
            [hl, -hw],
            [hl, hw],
            [-hl, hw],
            [-hl, -hw]  # Close the box
        ])
        
        # Rotation matrix
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        # Transform corners
        corners_world = corners_local @ R.T + np.array([x, y])
        
        ax.plot(corners_world[:, 0], corners_world[:, 1], 
                color=color, linewidth=config.box_linewidth, label=label)
        
        # Draw heading indicator
        heading = np.array([[0, 0], [hl, 0]]) @ R.T + np.array([x, y])
        ax.plot(heading[:, 0], heading[:, 1], color=color, linewidth=config.box_linewidth)
    
    # Draw GT boxes
    if gt_boxes:
        for i, box in enumerate(gt_boxes):
            label = 'GT' if i == 0 else None
            draw_rotated_box(
                ax, box['x'], box['y'], box['w'], box['l'], box['yaw'],
                config.box_gt_color, label
            )
    
    # Draw predicted boxes
    if pred_boxes:
        for i, box in enumerate(pred_boxes):
            label = 'Pred' if i == 0 else None
            draw_rotated_box(
                ax, box['x'], box['y'], box['w'], box['l'], box['yaw'],
                config.box_pred_color, label
            )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Mark ego
    ax.plot(0, 0, 'b*', markersize=15, label='Ego')
    ax.legend(loc='upper right')
    
    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or config.output_dir)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def visualize_fusion_comparison(
    lidar_features: torch.Tensor,
    camera_features: torch.Tensor,
    fused_features: torch.Tensor,
    title: str = "Fusion Comparison",
    save_path: Optional[str] = None,
    config: Optional[BEVVisConfig] = None
) -> plt.Figure:
    """
    Side-by-side visualization of LiDAR, Camera, and Fused features.
    
    Args:
        lidar_features: LiDAR BEV features
        camera_features: Camera BEV features
        fused_features: Fused BEV features
        title: Overall title
        save_path: Save path
        config: Visualization config
        
    Returns:
        matplotlib Figure
    """
    config = config or BEVVisConfig()
    
    # Convert to numpy and normalize
    lidar = normalize_features(tensor_to_numpy(lidar_features).mean(axis=0) 
                               if tensor_to_numpy(lidar_features).ndim > 2 
                               else tensor_to_numpy(lidar_features))
    camera = normalize_features(tensor_to_numpy(camera_features).mean(axis=0) 
                                if tensor_to_numpy(camera_features).ndim > 2 
                                else tensor_to_numpy(camera_features))
    fused = normalize_features(tensor_to_numpy(fused_features).mean(axis=0) 
                               if tensor_to_numpy(fused_features).ndim > 2 
                               else tensor_to_numpy(fused_features))
    
    while lidar.ndim > 2:
        lidar = lidar[0]
    while camera.ndim > 2:
        camera = camera[0]
    while fused.ndim > 2:
        fused = fused[0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=config.dpi)
    
    x_min, x_max, y_min, y_max = config.bev_range
    extent = [x_min, x_max, y_min, y_max]
    
    for ax, data, subtitle in zip(axes, [lidar, camera, fused], 
                                  ['LiDAR BEV', 'Camera BEV', 'Fused BEV']):
        im = ax.imshow(data, cmap=config.cmap_features, extent=extent, origin='lower')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(subtitle)
        ax.plot(0, 0, 'r*', markersize=10)
        plt.colorbar(im, ax=ax)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or config.output_dir)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def visualize_temporal_sequence(
    features_sequence: List[torch.Tensor],
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    config: Optional[BEVVisConfig] = None
) -> plt.Figure:
    """
    Visualize a sequence of BEV features (temporal).
    
    Args:
        features_sequence: List of feature tensors
        titles: Optional titles for each frame
        save_path: Save path
        config: Visualization config
        
    Returns:
        matplotlib Figure
    """
    config = config or BEVVisConfig()
    n_frames = len(features_sequence)
    
    fig, axes = plt.subplots(1, n_frames, figsize=(4 * n_frames, 4), dpi=config.dpi)
    if n_frames == 1:
        axes = [axes]
    
    x_min, x_max, y_min, y_max = config.bev_range
    
    for i, (ax, feat) in enumerate(zip(axes, features_sequence)):
        feat_np = tensor_to_numpy(feat)
        while feat_np.ndim > 2:
            feat_np = feat_np.mean(axis=0) if feat_np.ndim == 3 else feat_np[0]
        feat_np = normalize_features(feat_np)
        
        im = ax.imshow(
            feat_np,
            cmap=config.cmap_features,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower'
        )
        
        title = titles[i] if titles else f"Frame {i}"
        ax.set_title(title)
        ax.set_xlabel('X (m)')
        if i == 0:
            ax.set_ylabel('Y (m)')
        ax.plot(0, 0, 'r*', markersize=8)
    
    plt.tight_layout()
    
    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or config.output_dir)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig


def visualize_attention_map(
    attention: torch.Tensor,
    query_pos: Optional[Tuple[int, int]] = None,
    title: str = "Attention Map",
    save_path: Optional[str] = None,
    config: Optional[BEVVisConfig] = None
) -> plt.Figure:
    """
    Visualize attention weights.
    
    Args:
        attention: Attention tensor (H*W, H*W) or (num_heads, H*W, H*W)
        query_pos: If provided, show attention for this query position (y, x)
        title: Plot title
        save_path: Save path
        config: Visualization config
        
    Returns:
        matplotlib Figure
    """
    config = config or BEVVisConfig()
    attn = tensor_to_numpy(attention)
    
    # Average over heads if present
    if attn.ndim == 3:
        attn = attn.mean(axis=0)
    
    N = attn.shape[0]
    H = W = int(np.sqrt(N))
    
    if query_pos is not None:
        # Show attention for specific query
        qy, qx = query_pos
        query_idx = qy * W + qx
        attn_map = attn[query_idx].reshape(H, W)
        title = f"{title} (query at {query_pos})"
    else:
        # Show average attention
        attn_map = attn.mean(axis=0).reshape(H, W)
    
    fig, ax = plt.subplots(figsize=config.fig_size, dpi=config.dpi)
    
    x_min, x_max, y_min, y_max = config.bev_range
    im = ax.imshow(
        attn_map,
        cmap='viridis',
        extent=[x_min, x_max, y_min, y_max],
        origin='lower'
    )
    
    if query_pos is not None:
        # Mark query position
        qy, qx = query_pos
        # Convert to world coordinates
        qx_world = x_min + (qx + 0.5) * (x_max - x_min) / W
        qy_world = y_min + (qy + 0.5) * (y_max - y_min) / H
        ax.plot(qx_world, qy_world, 'r*', markersize=15, label='Query')
        ax.legend()
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Attention Weight')
    
    if save_path:
        ensure_output_dir(os.path.dirname(save_path) or config.output_dir)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
    return fig

