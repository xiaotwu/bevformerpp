"""
Paper-quality figure generation for BEVFormer++ visualization.

This module provides the main visualization API:
    visualize_bevformer_case() - Single entry point for all visualizations

Design principles:
1. Strict camera_only vs fusion semantics
2. No fake or duplicated tensors
3. Shared color scales for comparability
4. Publication-ready formatting

SEMANTIC SEPARATION:
- Row 1: INPUT DATA (Camera, LiDAR density, GT centers)
- Row 2: DETECTION HEAD OUTPUT (NOT BEV features!)
- Row 3: BEV REPRESENTATION (intermediate features)

This distinction is critical for correct interpretation.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Literal, List
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from .core import (
    normalize_for_display,
    compute_bev_energy,
    compute_shared_normalization,
    extract_nms_peaks,
    get_gt_centers,
    extract_visual_tensors,
    VisualTensors,
    Peak,
    compute_tensor_stats,
)
from .lidar_bev import project_lidar_to_bev, debug_lidar_bev, extract_lidar_points_np


# ==============================================================================
# DEBUG STATS UTILITIES
# ==============================================================================

def print_tensor_debug_stats(name: str, tensor: Optional[torch.Tensor]) -> Dict:
    """Print and return debug statistics for a tensor."""
    if tensor is None:
        print(f"  [{name}] N/A (not captured)")
        return {'available': False, 'reason': 'not captured'}
    
    stats = compute_tensor_stats(tensor)
    
    # Check for degeneracy
    dynamic_range = stats['p99'] - stats['p1']
    is_degenerate = dynamic_range < 1e-6
    
    print(f"  [{name}]")
    print(f"    shape: {stats['shape']}, dtype: {stats['dtype']}")
    print(f"    range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"    mean: {stats['mean']:.4f}, std: {stats['std']:.4f}")
    print(f"    percentiles: p1={stats['p1']:.4f}, p50={stats['p50']:.4f}, p99={stats['p99']:.4f}")
    
    if is_degenerate:
        print(f"    ⚠️ LOW DYNAMIC RANGE: p99-p1 = {dynamic_range:.2e}")
    
    stats['is_degenerate'] = is_degenerate
    stats['dynamic_range'] = dynamic_range
    stats['available'] = True
    return stats


# ==============================================================================
# INDIVIDUAL PANEL PLOTTING
# ==============================================================================

def _plot_camera_images(
    img: torch.Tensor,
    ax: plt.Axes,
    title: str = "Camera Views"
) -> None:
    """Plot multi-view camera images concatenated horizontally."""
    # Handle shapes: (B, T, N, C, H, W) or (B, N, C, H, W)
    t = img.detach()
    if t.dim() == 6:
        t = t[0, -1]  # Take last timestep: (N, C, H, W)
    elif t.dim() == 5:
        t = t[0]  # (N, C, H, W)
    else:
        ax.text(0.5, 0.5, f'Invalid shape: {img.shape}', 
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        ax.axis('off')
        return
    
    N = min(t.shape[0], 6)  # Max 6 cameras
    
    imgs = []
    for i in range(N):
        cam = t[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        # Normalize to [0, 1]
        if cam.max() > 1.0:
            cam = cam / 255.0
        cam = np.clip(cam, 0, 1)
        imgs.append(cam)
    
    concat = np.concatenate(imgs, axis=1)
    ax.imshow(concat)
    ax.set_title(title)
    ax.axis('off')


def _plot_lidar_density(
    lidar_points: torch.Tensor,
    ax: plt.Axes,
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
    title: str = "LiDAR BEV Density",
    lidar_mask: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Plot LiDAR point cloud as BEV density with log scale.

    This function now uses extract_lidar_points_np() for robust handling of
    various tensor shapes including (B, T, N, 4) sequences.

    Args:
        lidar_points: LiDAR points tensor of any supported shape
        ax: Matplotlib axes
        bev_range: (x_min, x_max, y_min, y_max) in meters
        title: Plot title
        lidar_mask: Optional mask tensor

    Returns:
        Diagnostic dict with point statistics.
    """
    # Use robust extraction for all input shapes
    try:
        pts_np = extract_lidar_points_np(
            lidar_points,
            lidar_mask=lidar_mask,
            take_t="last",
            take_b=0,
            require_cols=3
        )
    except ValueError as e:
        ax.text(0.5, 0.5, f'LiDAR Error:\n{str(e)[:50]}...',
                ha='center', va='center', transform=ax.transAxes, fontsize=9, color='red')
        ax.set_title(title)
        ax.axis('off')
        return {'error': str(e)}

    if len(pts_np) == 0:
        ax.text(0.5, 0.5, 'LiDAR: No valid points',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title(title)
        ax.axis('off')
        return {'num_points_in_range': 0}

    # Get diagnostics
    diag = debug_lidar_bev(
        pts_np,
        x_range=(bev_range[0], bev_range[1]),
        y_range=(bev_range[2], bev_range[3]),
    )

    # Project to BEV
    density = project_lidar_to_bev(
        pts_np,
        x_range=(bev_range[0], bev_range[1]),
        y_range=(bev_range[2], bev_range[3]),
    )

    # Log scale for better visualization
    density_log = np.log1p(density)

    # Clip to 99th percentile
    vmax = np.percentile(density_log, 99)
    if vmax <= 0:
        vmax = 1.0

    im = ax.imshow(
        density_log, origin='lower',
        extent=[bev_range[0], bev_range[1], bev_range[2], bev_range[3]],
        cmap='viridis', aspect='equal',
        vmin=0, vmax=vmax
    )

    # Annotations
    n_valid = diag['num_points_in_range']
    clip_pct = diag['fraction_clipped'] * 100
    ax.set_title(f"{title}\n({n_valid:,} pts, {clip_pct:.1f}% clipped)", fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    plt.colorbar(im, ax=ax, fraction=0.046, label='log(1+count)')

    return diag


def _plot_na_panel(ax: plt.Axes, title: str, reason: str) -> None:
    """Plot an N/A panel with explanation."""
    ax.text(0.5, 0.5, f'N/A\n({reason})', 
            ha='center', va='center', transform=ax.transAxes, 
            fontsize=11, color='gray')
    ax.set_title(title, fontsize=10)
    ax.set_facecolor('#f5f5f5')
    ax.axis('off')


def _plot_gt_heatmap(
    cls_targets: torch.Tensor,
    ax: plt.Axes,
    title: str = "GT Heatmap",
    shared_norm: Optional[Tuple[float, float]] = None,
) -> int:
    """
    Plot ground truth classification heatmap.
    
    GT is binary (hard_center mode), so we use a distinct visualization
    that clearly shows center locations.
    
    Returns number of GT centers.
    """
    t = cls_targets.detach()
    if t.dim() == 4:
        t = t[0]  # (C, H, W)
    
    # Max across classes for visualization
    h = t.max(dim=0)[0].cpu().numpy()  # (H, W)
    
    # Get GT centers
    centers = get_gt_centers(cls_targets)
    n_centers = len(centers)
    
    # Plot heatmap
    # For GT, always use [0, 1] scale since it's binary
    im = ax.imshow(h, origin='lower', cmap='hot', aspect='equal', vmin=0, vmax=1)
    
    # Mark centers
    if n_centers > 0:
        cy = [c[0] for c in centers]
        cx = [c[1] for c in centers]
        ax.scatter(cx, cy, c='cyan', s=60, marker='x', linewidths=2,
                  label=f'{n_centers} GT centers')
        ax.legend(loc='upper right', fontsize=8)
    
    ax.set_title(f"{title} ({n_centers} objects)", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    return n_centers


def _plot_pred_heatmap(
    pred_heatmap: torch.Tensor,
    ax: plt.Axes,
    title: str = "HEAD: Pred Heatmap",
    gt_centers: Optional[List[Tuple[int, int, int]]] = None,
) -> Dict:
    """
    Plot predicted classification heatmap with proper normalization.
    
    IMPORTANT: This is HEAD OUTPUT (after detection head), NOT a BEV feature!
    Always applies sigmoid and uses percentile normalization for proper visualization.
    """
    t = pred_heatmap.detach()
    if t.dim() == 4:
        t = t[0]  # (C, H, W)
    
    # Apply sigmoid
    h_prob = torch.sigmoid(t)
    
    # Max across classes
    h = h_prob.max(dim=0)[0].cpu().numpy()  # (H, W)
    
    # Percentile normalization for pred (not [0,1] since predictions may be sparse)
    vmin = np.percentile(h, 1)
    vmax = np.percentile(h, 99)
    if vmax <= vmin:
        vmax = 1.0
        vmin = 0.0
    
    im = ax.imshow(h, origin='lower', cmap='hot', aspect='equal', vmin=vmin, vmax=vmax)
    
    # Overlay GT centers if provided
    if gt_centers is not None and len(gt_centers) > 0:
        cy = [c[0] for c in gt_centers]
        cx = [c[1] for c in gt_centers]
        ax.scatter(cx, cy, c='cyan', s=40, marker='+', linewidths=1.5,
                  alpha=0.8, label='GT')
    
    # CLEAR LABELING: This is HEAD output, not BEV feature
    ax.set_title(f"{title}\n(sigmoid applied, p1-p99 scale)", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    stats = {
        'prob_max': float(h.max()),
        'prob_mean': float(h.mean()),
        'vmin': vmin,
        'vmax': vmax,
    }
    
    return stats


def _plot_pred_peaks(
    pred_heatmap: torch.Tensor,
    ax: plt.Axes,
    k: int = 50,
    radius: int = 3,
    thresh: float = 0.1,
    gt_centers: Optional[List[Tuple[int, int, int]]] = None,
    title: str = "HEAD: Pred + NMS Peaks",
) -> Dict:
    """
    Plot predicted heatmap with NMS peaks overlaid.
    
    IMPORTANT: This is HEAD OUTPUT visualization, NOT a BEV feature!
    
    Shows:
    - Heatmap as background
    - NMS-filtered peaks as circles
    - GT centers as crosses
    - Peak-to-GT distances (if GT available)
    """
    t = pred_heatmap.detach()
    if t.dim() == 4:
        t = t[0]  # (C, H, W)
    
    # Apply sigmoid and max across classes
    h_prob = torch.sigmoid(t)
    h = h_prob.max(dim=0)[0].cpu().numpy()
    
    # Plot background heatmap
    vmax = np.percentile(h, 99)
    im = ax.imshow(h, origin='lower', cmap='hot', aspect='equal', vmin=0, vmax=max(vmax, 0.1))
    
    # Extract NMS peaks
    peaks = extract_nms_peaks(pred_heatmap, k=k, radius=radius, thresh=thresh)
    n_peaks = len(peaks)
    
    # Plot peaks
    if n_peaks > 0:
        px = [p.x for p in peaks]
        py = [p.y for p in peaks]
        ps = [p.score for p in peaks]
        
        scatter = ax.scatter(px, py, c='lime', s=50, marker='o', 
                            edgecolors='white', linewidths=1.0, alpha=0.9,
                            label=f'{n_peaks} peaks')
        
        # Annotate top 5 peaks with scores
        for i, peak in enumerate(peaks[:5]):
            ax.annotate(f'{peak.score:.2f}', (peak.x, peak.y),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=7, color='white')
    
    # Plot GT centers
    if gt_centers is not None and len(gt_centers) > 0:
        cy = [c[0] for c in gt_centers]
        cx = [c[1] for c in gt_centers]
        ax.scatter(cx, cy, c='cyan', s=80, marker='x', linewidths=2,
                  label=f'{len(gt_centers)} GT')
    
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f"{title}\n(k={k}, r={radius}, t={thresh})", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Compute peak-to-GT distance if possible
    stats = {'n_peaks': n_peaks, 'peaks': peaks}
    if gt_centers is not None and len(gt_centers) > 0 and n_peaks > 0:
        distances = []
        for peak in peaks:
            min_dist = float('inf')
            for gt in gt_centers:
                d = np.sqrt((peak.y - gt[0])**2 + (peak.x - gt[1])**2)
                min_dist = min(min_dist, d)
            distances.append(min_dist)
        stats['mean_peak_to_gt_dist'] = float(np.mean(distances))
        stats['peaks_within_5px'] = sum(1 for d in distances if d < 5)
    
    return stats


def _plot_bev_feature(
    feature: Optional[torch.Tensor],
    ax: plt.Axes,
    title: str,
    vmin: float,
    vmax: float,
    na_reason: str = "Not produced",
    energy_method: str = 'l2_norm',
) -> Optional[Dict]:
    """
    Plot BEV feature map with shared normalization.
    
    Uses L2 norm (energy) across channels to create interpretable visualization.
    If feature is None, shows N/A panel with reason.
    
    Returns stats dict or None if N/A.
    """
    if feature is None:
        _plot_na_panel(ax, title, na_reason)
        return None
    
    # Compute energy (channel aggregation)
    energy = compute_bev_energy(feature, method=energy_method)
    
    # Check for degeneracy
    energy_range = energy.max() - energy.min()
    is_degenerate = energy_range < 1e-6
    
    # Normalize using shared scale
    if vmax > vmin:
        energy_norm = (energy - vmin) / (vmax - vmin + 1e-8)
        energy_norm = np.clip(energy_norm, 0, 1)
    else:
        energy_norm = np.full_like(energy, 0.5)
    
    im = ax.imshow(energy_norm, origin='lower', cmap='viridis', aspect='equal', vmin=0, vmax=1)
    
    # Add warning if degenerate
    subtitle = f"(L2 norm, shared scale)"
    if is_degenerate:
        subtitle = f"⚠️ LOW DYNAMIC RANGE"
        ax.text(0.5, 0.02, "Nearly constant!", transform=ax.transAxes,
               ha='center', fontsize=8, color='red', 
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_title(f"{title}\n{subtitle}", fontsize=9)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='energy')
    
    return {
        'energy_min': float(energy.min()),
        'energy_max': float(energy.max()),
        'energy_mean': float(energy.mean()),
        'is_degenerate': is_degenerate,
    }


# ==============================================================================
# MAIN VISUALIZATION API
# ==============================================================================

def visualize_bevformer_case(
    batch: Dict,
    preds: Dict,
    tensors: Dict,
    case_name: str,
    output_dir: Path,
    mode: Literal['camera_only', 'fusion'],
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
    peak_k: int = 50,
    peak_radius: int = 3,
    peak_thresh: float = 0.1,
    energy_method: str = 'l2_norm',
    print_debug_stats: bool = True,
) -> plt.Figure:
    """
    Produce a paper-ready BEVFormer++ visualization figure.
    
    This is the SINGLE entry point for all visualization.
    
    SEMANTIC LAYOUT (3x3 grid):
        Row 1 [INPUT DATA]:
            Camera images | LiDAR density | GT centers
        Row 2 [DETECTION HEAD OUTPUT]:
            GT heatmap | Pred heatmap (HEAD) | Pred peaks (HEAD)
        Row 3 [BEV REPRESENTATION]:
            Camera BEV | LiDAR BEV | Fused BEV
    
    CRITICAL: Row 2 shows HEAD outputs (after detection head, NOT BEV features).
              Row 3 shows intermediate BEV features (before detection head).
    
    Mode semantics:
        camera_only: LiDAR and Fused panels show "N/A (camera-only mode)"
        fusion: All panels should have content (or "N/A" if hook failed)
    
    Args:
        batch: Batch dict with 'img', 'lidar_points' (if fusion), 'cls_targets'
        preds: Model predictions dict
        tensors: Captured intermediate tensors from forward hooks
        case_name: Name for this case (e.g., 'CameraOnly', 'Fusion')
        output_dir: Directory to save figure
        mode: 'camera_only' or 'fusion' - determines panel semantics
        bev_range: (x_min, x_max, y_min, y_max) in meters
        peak_k: Max peaks for NMS extraction
        peak_radius: NMS radius in pixels
        peak_thresh: Peak score threshold
        energy_method: Method for BEV energy computation ('l2_norm' recommended)
        print_debug_stats: Whether to print detailed tensor statistics
    
    Returns:
        matplotlib Figure object
    """
    # Print debug stats for all tensors
    if print_debug_stats:
        print(f"\n=== Debug Stats: {case_name} ({mode}) ===")
        print(f"Captured tensor keys: {list(tensors.keys())}")
        print(f"Prediction keys: {list(preds.keys())}")
    
    # Extract tensors with strict semantics
    vis_tensors = extract_visual_tensors(preds, tensors, mode)
    
    if print_debug_stats:
        print(f"\n--- BEV Feature Tensors ---")
        camera_stats = print_tensor_debug_stats("camera_bev", vis_tensors.camera_bev)
        lidar_stats = print_tensor_debug_stats("lidar_bev", vis_tensors.lidar_bev)
        fused_stats = print_tensor_debug_stats("fused_bev", vis_tensors.fused_bev)
        print(f"\n--- Prediction Tensors ---")
        pred_stats = print_tensor_debug_stats("pred_heatmap", vis_tensors.pred_heatmap)
    
    # Get GT centers for overlay
    gt_centers = get_gt_centers(batch['cls_targets'])
    
    # Create figure
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.30, wspace=0.25)
    
    mode_label = "Camera-Only" if mode == 'camera_only' else "Fusion (Camera + LiDAR)"
    fig.suptitle(f'BEVFormer++ Visualization: {case_name}\nMode: {mode_label}', 
                 fontsize=14, fontweight='bold')
    
    diagnostics = {'case': case_name, 'mode': mode}
    
    # ==================== ROW 1: INPUT DATA ====================
    ax_cam = fig.add_subplot(gs[0, 0])
    _plot_camera_images(batch['img'], ax_cam, "INPUT: Camera Views")
    
    ax_lidar = fig.add_subplot(gs[0, 1])
    if mode == 'fusion' and 'lidar_points' in batch:
        # Get mask if available (supports both 'lidar_mask' and 'lidar_mask_seq' keys)
        lidar_mask = batch.get('lidar_mask') or batch.get('lidar_mask_seq')
        diag = _plot_lidar_density(
            batch['lidar_points'], ax_lidar, bev_range, "INPUT: LiDAR Density",
            lidar_mask=lidar_mask
        )
        diagnostics['lidar'] = diag
    else:
        reason = "camera-only mode" if mode == 'camera_only' else "LiDAR not in batch"
        _plot_na_panel(ax_lidar, "INPUT: LiDAR Density", reason)
    
    ax_gt = fig.add_subplot(gs[0, 2])
    n_gt = _plot_gt_heatmap(batch['cls_targets'], ax_gt, "INPUT: GT Centers")
    diagnostics['n_gt_objects'] = n_gt
    
    # ==================== ROW 2: DETECTION HEAD OUTPUT ====================
    # IMPORTANT: These are HEAD outputs, NOT intermediate BEV features!
    ax_gt_hm = fig.add_subplot(gs[1, 0])
    _plot_gt_heatmap(batch['cls_targets'], ax_gt_hm, "TARGET: GT Heatmap")
    
    ax_pred = fig.add_subplot(gs[1, 1])
    ax_peaks = fig.add_subplot(gs[1, 2])
    
    if vis_tensors.pred_heatmap is not None:
        pred_stats = _plot_pred_heatmap(
            vis_tensors.pred_heatmap, ax_pred, "HEAD OUTPUT: Pred Heatmap", gt_centers
        )
        diagnostics['pred_heatmap'] = pred_stats
        
        peak_stats = _plot_pred_peaks(
            vis_tensors.pred_heatmap, ax_peaks,
            k=peak_k, radius=peak_radius, thresh=peak_thresh,
            gt_centers=gt_centers, title="HEAD OUTPUT: Pred + NMS Peaks"
        )
        diagnostics['peaks'] = {
            'n_peaks': peak_stats['n_peaks'],
            'mean_dist_to_gt': peak_stats.get('mean_peak_to_gt_dist'),
            'peaks_within_5px': peak_stats.get('peaks_within_5px'),
        }
    else:
        _plot_na_panel(ax_pred, "HEAD OUTPUT: Pred Heatmap", "No predictions")
        _plot_na_panel(ax_peaks, "HEAD OUTPUT: Pred + NMS Peaks", "No predictions")
    
    # ==================== ROW 3: BEV REPRESENTATION ====================
    # These are INTERMEDIATE features, NOT head outputs!
    
    # Compute shared normalization across all available features
    features = [vis_tensors.camera_bev, vis_tensors.lidar_bev, vis_tensors.fused_bev]
    features_valid = [f for f in features if f is not None]
    
    if len(features_valid) > 0:
        vmin, vmax = compute_shared_normalization(features_valid, method=energy_method)
    else:
        vmin, vmax = 0.0, 1.0
    
    diagnostics['feature_scale'] = {'vmin': vmin, 'vmax': vmax}
    diagnostics['available_features'] = vis_tensors.available_features()
    
    ax_cam_bev = fig.add_subplot(gs[2, 0])
    cam_bev_stats = _plot_bev_feature(
        vis_tensors.camera_bev, ax_cam_bev, "BEV REPR: Camera",
        vmin, vmax, na_reason="Hook not captured", energy_method=energy_method
    )
    if cam_bev_stats is not None:
        diagnostics['camera_bev_stats'] = cam_bev_stats
    
    ax_lidar_bev = fig.add_subplot(gs[2, 1])
    lidar_na = "camera-only mode" if mode == 'camera_only' else "Hook not captured"
    lidar_bev_stats = _plot_bev_feature(
        vis_tensors.lidar_bev, ax_lidar_bev, "BEV REPR: LiDAR",
        vmin, vmax, na_reason=lidar_na, energy_method=energy_method
    )
    if lidar_bev_stats is not None:
        diagnostics['lidar_bev_stats'] = lidar_bev_stats
    
    ax_fused_bev = fig.add_subplot(gs[2, 2])
    fused_na = "camera-only mode" if mode == 'camera_only' else "Hook not captured"
    fused_bev_stats = _plot_bev_feature(
        vis_tensors.fused_bev, ax_fused_bev, "BEV REPR: Fused",
        vmin, vmax, na_reason=fused_na, energy_method=energy_method
    )
    if fused_bev_stats is not None:
        diagnostics['fused_bev_stats'] = fused_bev_stats
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = output_dir / f"fig_{case_name}.png"
    pdf_path = output_dir / f"fig_{case_name}.pdf"
    
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    print(f"✓ Saved: {png_path}")
    print(f"✓ Saved: {pdf_path}")
    
    # Print diagnostics summary
    print(f"\n=== Diagnostics Summary: {case_name} ({mode}) ===")
    print(f"  GT objects: {diagnostics['n_gt_objects']}")
    if 'lidar' in diagnostics:
        print(f"  LiDAR points in range: {diagnostics['lidar']['num_points_in_range']:,}")
    if 'peaks' in diagnostics:
        print(f"  Detected peaks: {diagnostics['peaks']['n_peaks']}")
        if diagnostics['peaks']['mean_dist_to_gt'] is not None:
            print(f"  Mean peak-to-GT distance: {diagnostics['peaks']['mean_dist_to_gt']:.1f} px")
            print(f"  Peaks within 5px of GT: {diagnostics['peaks']['peaks_within_5px']}")
    print(f"  Available BEV features: {diagnostics['available_features']}")
    print(f"  Feature energy scale: [{vmin:.4f}, {vmax:.4f}]")
    
    return fig


def create_side_by_side_comparison(
    batch_camera: Dict,
    preds_camera: Dict,
    tensors_camera: Dict,
    batch_fusion: Dict,
    preds_fusion: Dict,
    tensors_fusion: Dict,
    output_dir: Path,
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Create side-by-side comparison figures for camera-only and fusion.
    
    Returns both figures for inspection.
    """
    fig_cam = visualize_bevformer_case(
        batch=batch_camera,
        preds=preds_camera,
        tensors=tensors_camera,
        case_name='CameraOnly',
        output_dir=output_dir,
        mode='camera_only',
        bev_range=bev_range,
    )
    
    fig_fusion = visualize_bevformer_case(
        batch=batch_fusion,
        preds=preds_fusion,
        tensors=tensors_fusion,
        case_name='Fusion',
        output_dir=output_dir,
        mode='fusion',
        bev_range=bev_range,
    )
    
    return fig_cam, fig_fusion
