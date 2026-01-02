"""
Heatmap peak extraction utilities with proper local maxima detection (NMS-like).

This module provides:
- Local maxima extraction using max pooling
- Top-K peak extraction with NMS
- Heatmap summary statistics
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


def extract_local_maxima(
    heatmap: torch.Tensor,
    kernel_size: int = 3,
    threshold: float = 0.0,
) -> torch.Tensor:
    """
    Extract local maxima from heatmap using max pooling (NMS-like).
    
    A pixel is a local maximum if it equals the max in its kernel_size neighborhood.
    
    Args:
        heatmap: (B, C, H, W) or (C, H, W) or (H, W) tensor
        kernel_size: Size of local neighborhood for maxima detection
        threshold: Minimum value threshold for peaks
    
    Returns:
        peaks: Same shape as heatmap, with local maxima retained and others zeroed
    """
    # Handle various input shapes
    original_shape = heatmap.shape
    squeeze_dims = []
    
    if heatmap.dim() == 2:  # (H, W)
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        squeeze_dims = [0, 1]
    elif heatmap.dim() == 3:  # (C, H, W)
        heatmap = heatmap.unsqueeze(0)  # (1, C, H, W)
        squeeze_dims = [0]
    
    # Apply max pooling
    padding = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=padding)
    
    # Find local maxima (where value equals local max)
    is_local_max = (heatmap == hmax).float()
    
    # Apply threshold
    above_threshold = (heatmap >= threshold).float()
    
    # Combine: local maxima above threshold
    peaks = heatmap * is_local_max * above_threshold
    
    # Restore original shape
    for dim in reversed(squeeze_dims):
        peaks = peaks.squeeze(dim)
    
    return peaks


def extract_topk_peaks_nms(
    heatmap: torch.Tensor,
    k: int = 100,
    kernel_size: int = 3,
    threshold: float = 0.1,
    per_class: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract top-K peaks with NMS-like local maxima filtering.
    
    This prevents clustering of peaks at a single high-activation region
    by requiring peaks to be local maxima in their neighborhood.
    
    Args:
        heatmap: (B, C, H, W) heatmap tensor
        k: Maximum number of peaks per class (or total if per_class=False)
        kernel_size: NMS kernel size
        threshold: Minimum score threshold
        per_class: If True, extract k peaks per class. If False, extract k peaks total.
    
    Returns:
        coords: (N, 4) tensor with [batch_idx, class_idx, y, x] for each peak
        scores: (N,) tensor with peak scores
        peak_map: (B, C, H, W) tensor with only peaks retained
        peak_counts: (B, C) tensor with number of peaks per batch/class
    """
    B, C, H, W = heatmap.shape
    device = heatmap.device
    
    # First apply local maxima detection
    peaks = extract_local_maxima(heatmap, kernel_size, threshold)
    
    all_coords = []
    all_scores = []
    peak_counts = torch.zeros((B, C), dtype=torch.long, device=device)
    
    for b in range(B):
        for c in range(C):
            cls_peaks = peaks[b, c]  # (H, W)
            
            # Flatten and find non-zero peaks
            flat = cls_peaks.view(-1)
            
            # Get indices of non-zero elements
            nonzero_mask = flat > 0
            nonzero_count = nonzero_mask.sum().item()
            
            if nonzero_count == 0:
                continue
            
            # Get top-k among non-zero
            actual_k = min(k, nonzero_count)
            topk_scores, topk_flat_idx = torch.topk(flat, actual_k)
            
            # Filter by threshold (redundant but safe)
            valid = topk_scores > threshold
            topk_scores = topk_scores[valid]
            topk_flat_idx = topk_flat_idx[valid]
            
            if len(topk_scores) == 0:
                continue
            
            # Convert flat indices to 2D
            y_idx = topk_flat_idx // W
            x_idx = topk_flat_idx % W
            
            # Create coordinate tensor [batch, class, y, x]
            n_peaks = len(topk_scores)
            coords = torch.stack([
                torch.full((n_peaks,), b, dtype=torch.long, device=device),
                torch.full((n_peaks,), c, dtype=torch.long, device=device),
                y_idx,
                x_idx,
            ], dim=1)
            
            all_coords.append(coords)
            all_scores.append(topk_scores)
            peak_counts[b, c] = n_peaks
    
    # Concatenate results
    if len(all_coords) > 0:
        coords = torch.cat(all_coords, dim=0)
        scores = torch.cat(all_scores, dim=0)
    else:
        coords = torch.zeros((0, 4), dtype=torch.long, device=device)
        scores = torch.zeros((0,), dtype=heatmap.dtype, device=device)
    
    return coords, scores, peaks, peak_counts


def summarize_heatmap(
    heatmap: torch.Tensor,
    apply_sigmoid: bool = False,
) -> Dict:
    """
    Compute summary statistics for a heatmap.
    
    Args:
        heatmap: (B, C, H, W) tensor of logits or probabilities
        apply_sigmoid: If True, apply sigmoid before computing stats
    
    Returns:
        Dictionary with:
            - min, max, mean, std: Basic statistics
            - num_above_0.5: Count of values > 0.5 (after sigmoid if applied)
            - unique_values_sample: Sample of unique values (for checking target discretization)
            - shape: Tensor shape
    """
    h = heatmap.detach()
    
    if apply_sigmoid:
        h = torch.sigmoid(h)
    
    h_flat = h.view(-1)
    
    stats = {
        'shape': list(heatmap.shape),
        'min': float(h_flat.min()),
        'max': float(h_flat.max()),
        'mean': float(h_flat.mean()),
        'std': float(h_flat.std()),
        'num_above_0.5': int((h_flat > 0.5).sum()),
        'num_above_0.1': int((h_flat > 0.1).sum()),
        'num_nonzero': int((h_flat != 0).sum()),
    }
    
    # Sample unique values (for target heatmaps this reveals gaussian vs hard_center)
    unique = torch.unique(h_flat)
    if len(unique) <= 20:
        stats['unique_values'] = [float(v) for v in unique]
    else:
        stats['unique_values_sample'] = [float(v) for v in unique[:10]] + ['...'] + [float(v) for v in unique[-5:]]
        stats['num_unique'] = len(unique)
    
    return stats


def visualize_peaks_on_heatmap(
    heatmap: torch.Tensor,
    ax,
    k: int = 50,
    kernel_size: int = 3,
    threshold: float = 0.1,
    apply_sigmoid: bool = True,
    title: str = "Top-K Peaks (NMS)",
    cmap: str = 'hot',
    annotate_top_n: int = 5,
) -> Dict:
    """
    Plot heatmap with top-K NMS peaks overlaid.
    
    Args:
        heatmap: (B, C, H, W) or similar tensor
        ax: Matplotlib axes
        k: Number of peaks to extract
        kernel_size: NMS kernel size
        threshold: Peak threshold
        apply_sigmoid: Apply sigmoid to heatmap
        title: Plot title
        cmap: Colormap
        annotate_top_n: Number of top peaks to annotate with scores
    
    Returns:
        Dictionary with peak statistics
    """
    import matplotlib.pyplot as plt
    
    # Normalize shape to (B, C, H, W)
    h = heatmap.detach()
    if h.dim() == 2:
        h = h.unsqueeze(0).unsqueeze(0)
    elif h.dim() == 3:
        h = h.unsqueeze(0)
    
    if apply_sigmoid:
        h = torch.sigmoid(h)
    
    # Max over classes for visualization
    h_vis = h[0].max(dim=0)[0].cpu().numpy()  # (H, W)
    
    # Extract peaks
    coords, scores, _, counts = extract_topk_peaks_nms(h, k, kernel_size, threshold)
    
    # Plot heatmap
    im = ax.imshow(h_vis, origin='lower', cmap=cmap, aspect='equal', vmin=0, vmax=1)
    
    if len(coords) > 0:
        # Filter to batch 0
        mask = coords[:, 0] == 0
        peak_y = coords[mask, 2].cpu().numpy()
        peak_x = coords[mask, 3].cpu().numpy()
        peak_scores = scores[mask].cpu().numpy()
        
        # Plot peaks
        ax.scatter(peak_x, peak_y, c='cyan', s=30, marker='o', 
                   edgecolors='white', linewidths=0.5, alpha=0.8)
        
        # Annotate top peaks
        if annotate_top_n > 0 and len(peak_scores) > 0:
            sort_idx = np.argsort(peak_scores)[::-1]
            for i in sort_idx[:annotate_top_n]:
                ax.annotate(f'{peak_scores[i]:.2f}', 
                           (peak_x[i], peak_y[i]),
                           fontsize=7, color='white',
                           xytext=(3, 3), textcoords='offset points')
    
    total_peaks = int(counts.sum())
    ax.set_title(f"{title}\n({total_peaks} peaks, k={k})", fontsize=10)
    
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    return {
        'num_peaks': total_peaks,
        'peak_counts_per_class': counts[0].cpu().tolist() if counts.shape[0] > 0 else [],
    }



