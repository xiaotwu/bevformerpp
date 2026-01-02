"""
Paper-ready figure generation for BEVFormer++ visualization.

This module creates publication-quality comparison figures with:
- Proper LiDAR BEV density visualization
- NMS-based peak extraction
- Shared feature scaling across modalities
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt

from .lidar_bev import plot_lidar_bev_density, debug_lidar_bev
from .heatmap_peaks import (
    extract_topk_peaks_nms, summarize_heatmap, visualize_peaks_on_heatmap
)
from .feature_plots import (
    compute_shared_scale, plot_bev_feature_robust, compute_feature_metrics
)


def _first_not_none(*args):
    """Return first non-None argument."""
    for x in args:
        if x is not None:
            return x
    return None


def _plot_camera_grid(img_tensor: torch.Tensor, ax, title: str = "Camera Views"):
    """Plot multi-view camera images."""
    # Handle shapes: (B, T, N, C, H, W) or (B, N, C, H, W)
    if img_tensor.dim() == 6:
        img = img_tensor[0, -1]  # (N, C, H, W)
    elif img_tensor.dim() == 5:
        img = img_tensor[0]  # (N, C, H, W)
    else:
        ax.text(0.5, 0.5, 'Invalid image shape', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    
    n_cams = min(img.shape[0], 6)
    
    imgs = []
    for i in range(n_cams):
        cam_img = img[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        if cam_img.max() <= 1.0:
            cam_img = np.clip(cam_img, 0, 1)
        else:
            cam_img = np.clip(cam_img / 255.0, 0, 1)
        imgs.append(cam_img)
    
    concat = np.concatenate(imgs, axis=1)
    ax.imshow(concat)
    ax.set_title(title)
    ax.axis('off')


def _plot_gt_heatmap(cls_targets: torch.Tensor, ax, title: str = "GT Heatmap"):
    """Plot ground truth classification heatmap."""
    if cls_targets.dim() == 4:
        h = cls_targets[0].max(dim=0)[0]
    elif cls_targets.dim() == 3:
        h = cls_targets.max(dim=0)[0]
    else:
        h = cls_targets
    
    h = h.cpu().numpy()
    
    im = ax.imshow(h, origin='lower', cmap='hot', aspect='equal', vmin=0, vmax=1)
    
    # Mark GT centers
    cy, cx = np.where(h > 0.5)
    if len(cx) > 0:
        ax.scatter(cx, cy, c='cyan', s=50, marker='x', label=f'GT ({len(cx)})')
        ax.legend(loc='upper right', fontsize=8)
    
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)


def _infer_feature_keys(tensors: Dict) -> Dict:
    """Heuristically find BEV feature tensors by name patterns."""
    result = {'camera_bev': None, 'lidar_bev': None, 'fused_bev': None}
    
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() != 4:
            continue
        
        name_lower = name.lower()
        
        if any(p in name_lower for p in ['camera_bev', 'bev_cam', 'cam_bev', 'camera_encoder']):
            result['camera_bev'] = tensor
        elif any(p in name_lower for p in ['lidar_bev', 'bev_lidar', 'lidar_encoder', 'pillar']):
            result['lidar_bev'] = tensor
        elif any(p in name_lower for p in ['fused', 'fusion', 'merged']):
            result['fused_bev'] = tensor
        elif 'bev' in name_lower and result['fused_bev'] is None:
            result['fused_bev'] = tensor
    
    return result


def create_comparison_figure_v2(
    batch: Dict,
    preds: Dict,
    tensors: Dict,
    case_name: str,
    output_dir: Path,
    use_fusion: bool = False,
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
    peak_k: int = 50,
    peak_threshold: float = 0.1,
) -> Tuple[plt.Figure, Dict]:
    """
    Create paper-style 3x3 comparison figure with robust visualization.
    
    Layout:
        Row 1: Camera | LiDAR BEV | GT boxes
        Row 2: GT heatmap | Pred heatmap | Pred + TopK peaks
        Row 3: Camera BEV feat | LiDAR BEV feat | Fused BEV feat
    
    Args:
        batch: Batch dict with img, lidar_points, cls_targets, etc.
        preds: Model predictions dict
        tensors: Captured intermediate tensors
        case_name: Name for this case (e.g., 'CaseA_CameraOnly')
        output_dir: Directory to save figure
        use_fusion: Whether this is a fusion model
        bev_range: (x_min, x_max, y_min, y_max) for LiDAR visualization
        peak_k: Number of peaks to extract
        peak_threshold: Peak detection threshold
    
    Returns:
        fig: Matplotlib figure
        diag: Dictionary with diagnostic info
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'BEVFormer++ Visualization: {case_name}', fontsize=14, fontweight='bold')
    
    diag = {'case': case_name, 'use_fusion': use_fusion}
    
    # ========== Row 1: Inputs ==========
    _plot_camera_grid(batch['img'], axes[0, 0], title='Camera Views')
    
    # LiDAR BEV with proper projection
    if use_fusion and 'lidar_points' in batch:
        lidar_pts = batch['lidar_points']
        _, lidar_diag = plot_lidar_bev_density(
            lidar_pts, axes[0, 1],
            x_range=(bev_range[0], bev_range[1]),
            y_range=(bev_range[2], bev_range[3]),
            title='LiDAR BEV Density',
            log_scale=True,
            show_scatter=False,
        )
        diag['lidar'] = lidar_diag
    else:
        axes[0, 1].text(0.5, 0.5, 'LiDAR N/A\n(Camera-only)', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('LiDAR BEV')
        axes[0, 1].axis('off')
    
    # GT boxes/centers
    _plot_gt_heatmap(batch['cls_targets'], axes[0, 2], title='GT Centers')
    
    # ========== Row 2: Predictions ==========
    # GT heatmap
    _plot_gt_heatmap(batch['cls_targets'], axes[1, 0], title='GT Heatmap')
    
    # Get predicted heatmap
    pred_heatmap = None
    for key in ['cls_scores', 'cls_pred', 'heatmap', 'cls', 'output']:
        if key in preds:
            pred_heatmap = preds[key]
            break
    
    if pred_heatmap is not None:
        # Log heatmap stats
        diag['pred_heatmap_stats'] = summarize_heatmap(pred_heatmap, apply_sigmoid=False)
        diag['pred_heatmap_stats_sigmoid'] = summarize_heatmap(pred_heatmap, apply_sigmoid=True)
        
        # Pred heatmap (sigmoid)
        h_vis = pred_heatmap.detach()
        if h_vis.dim() == 4:
            h_vis = h_vis[0].max(dim=0)[0]
        h_vis = torch.sigmoid(h_vis).cpu().numpy()
        
        im1 = axes[1, 1].imshow(h_vis, origin='lower', cmap='hot', aspect='equal', vmin=0, vmax=1)
        axes[1, 1].set_title('Pred Heatmap (sigmoid)')
        plt.colorbar(im1, ax=axes[1, 1], fraction=0.046)
        
        # Top-K peaks with NMS
        peak_diag = visualize_peaks_on_heatmap(
            pred_heatmap, axes[1, 2],
            k=peak_k, threshold=peak_threshold,
            apply_sigmoid=True,
            title='Pred + Top-K Peaks (NMS)'
        )
        diag['peaks'] = peak_diag
    else:
        axes[1, 1].text(0.5, 0.5, 'Pred heatmap N/A', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Pred Heatmap')
        axes[1, 2].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Top-K Peaks')
    
    # ========== Row 3: BEV Features ==========
    bev_keys = _infer_feature_keys(tensors)
    
    # Also check for bev_features from camera-only forward
    cam_bev = _first_not_none(bev_keys.get('camera_bev'), tensors.get('bev_features'))
    lidar_bev = bev_keys.get('lidar_bev')
    fused_bev = bev_keys.get('fused_bev')
    
    # Compute shared scale across all features
    features_for_scale = [f for f in [cam_bev, lidar_bev, fused_bev] if f is not None]
    if features_for_scale:
        vmin, vmax = compute_shared_scale(features_for_scale)
    else:
        vmin, vmax = 0, 1
    
    diag['feature_scale'] = {'vmin': vmin, 'vmax': vmax}
    
    # Plot with shared scale
    plot_bev_feature_robust(cam_bev, axes[2, 0], 'Camera BEV', vmin=vmin, vmax=vmax, 
                           na_message='Camera BEV N/A')
    plot_bev_feature_robust(lidar_bev, axes[2, 1], 'LiDAR BEV', vmin=vmin, vmax=vmax,
                           na_message='LiDAR BEV N/A\n(Camera-only)' if not use_fusion else 'LiDAR BEV N/A')
    plot_bev_feature_robust(fused_bev, axes[2, 2], 'Fused BEV', vmin=vmin, vmax=vmax,
                           na_message='Fused BEV N/A')
    
    # Compute feature metrics
    diag['feature_metrics'] = compute_feature_metrics(cam_bev, lidar_bev, fused_bev)
    
    plt.tight_layout()
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"fig_sample_{case_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    
    # Also save PDF for paper
    pdf_path = output_dir / f"fig_sample_{case_name}.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    
    return fig, diag


def create_paper_figure(
    case_name: str,
    batch: Dict,
    preds: Dict,
    tensors: Dict,
    out_dir: Path,
    config: Optional[Dict] = None,
) -> Tuple[plt.Figure, Dict]:
    """
    Convenience wrapper for create_comparison_figure_v2.
    
    Args:
        case_name: Name of this case
        batch: Batch data
        preds: Model predictions
        tensors: Intermediate tensors
        out_dir: Output directory
        config: Optional config dict to extract BEV range, etc.
    
    Returns:
        fig, diagnostics
    """
    # Extract BEV range from config if available
    bev_range = (-51.2, 51.2, -51.2, 51.2)
    if config:
        bev_grid = config.get('bev_grid', {})
        if bev_grid:
            bev_range = (
                bev_grid.get('x_min', -51.2),
                bev_grid.get('x_max', 51.2),
                bev_grid.get('y_min', -51.2),
                bev_grid.get('y_max', 51.2),
            )
    
    use_fusion = False
    if config:
        use_fusion = config.get('use_fusion', False) or config.get('use_lidar', False)
    elif 'lidar_points' in batch:
        use_fusion = True
    
    return create_comparison_figure_v2(
        batch=batch,
        preds=preds,
        tensors=tensors,
        case_name=case_name,
        output_dir=out_dir,
        use_fusion=use_fusion,
        bev_range=bev_range,
    )



