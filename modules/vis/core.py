"""
Core visualization utilities for BEVFormer++ with strict semantic correctness.

This module provides:
- Robust normalization utilities
- BEV energy computation
- Proper NMS-based peak extraction
- Clean tensor extraction interface

Design Principles:
1. NO tensor truthiness checks (if tensor: is forbidden)
2. Explicit None checks only
3. Mode-aware semantics (camera_only vs fusion)
4. Publication-ready output
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Literal, Union
from dataclasses import dataclass


# ==============================================================================
# NORMALIZATION UTILITIES
# ==============================================================================

def normalize_for_display(
    tensor: torch.Tensor,
    pmin: float = 1.0,
    pmax: float = 99.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize tensor for display using percentile-based clipping.
    
    This is the ONLY normalization function to use for BEV features.
    It ensures:
    - Outliers don't dominate the color scale
    - Visual contrast is maximized
    - Results are comparable across different feature maps
    
    Args:
        tensor: Input tensor (any shape, will be flattened for percentile computation)
        pmin: Lower percentile for clipping (default: 1%)
        pmax: Upper percentile for clipping (default: 99%)
        eps: Small value to avoid division by zero
    
    Returns:
        Normalized numpy array in [0, 1]
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)
    
    # Flatten for percentile computation
    flat = arr.flatten()
    
    # Compute percentiles
    vmin = np.percentile(flat, pmin)
    vmax = np.percentile(flat, pmax)
    
    # Clip and normalize
    if vmax - vmin < eps:
        # Constant tensor - return zeros or mid-gray
        return np.full_like(arr, 0.5, dtype=np.float32)
    
    normalized = (arr - vmin) / (vmax - vmin + eps)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def compute_bev_energy(
    tensor: torch.Tensor,
    method: Literal['abs_mean', 'l2_norm', 'channel_mean'] = 'abs_mean'
) -> np.ndarray:
    """
    Compute per-pixel energy from BEV feature tensor.
    
    This aggregates channel information into a single 2D map that represents
    the "activity" or "energy" at each BEV location.
    
    Args:
        tensor: (B, C, H, W) or (C, H, W) feature tensor
        method: Aggregation method:
            - 'abs_mean': mean(abs(channels)) - shows overall activation magnitude
            - 'l2_norm': sqrt(sum(channels^2)) - emphasizes strong activations
            - 'channel_mean': mean(channels) - can cancel positive/negative
    
    Returns:
        (H, W) numpy array
    """
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu()
    else:
        t = torch.from_numpy(tensor)
    
    # Handle batch dimension
    if t.dim() == 4:
        t = t[0]  # (C, H, W)
    
    if t.dim() != 3:
        raise ValueError(f"Expected 3D or 4D tensor, got {t.dim()}D")
    
    if method == 'abs_mean':
        energy = t.abs().mean(dim=0)
    elif method == 'l2_norm':
        energy = torch.sqrt((t ** 2).sum(dim=0))
    elif method == 'channel_mean':
        energy = t.mean(dim=0)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return energy.numpy()


def compute_shared_normalization(
    tensors: List[Optional[torch.Tensor]],
    pmin: float = 1.0,
    pmax: float = 99.0,
    method: Literal['abs_mean', 'l2_norm', 'channel_mean'] = 'l2_norm',
) -> Tuple[float, float]:
    """
    Compute shared vmin/vmax across multiple feature tensors.
    
    This ensures that multiple BEV plots can be visually compared:
    the same color represents the same value across all plots.
    
    Args:
        tensors: List of tensors (None entries are skipped)
        pmin, pmax: Percentiles for normalization
        method: Energy computation method (l2_norm recommended for interpretability)
    
    Returns:
        (vmin, vmax) tuple
    """
    all_values = []
    
    for t in tensors:
        if t is None:
            continue
        
        energy = compute_bev_energy(t, method=method)
        all_values.append(energy.flatten())
    
    if len(all_values) == 0:
        return 0.0, 1.0
    
    combined = np.concatenate(all_values)
    vmin = float(np.percentile(combined, pmin))
    vmax = float(np.percentile(combined, pmax))
    
    if vmax <= vmin:
        vmax = vmin + 1.0
    
    return vmin, vmax


def compute_tensor_stats(tensor: torch.Tensor) -> Dict:
    """
    Compute comprehensive statistics for debugging.
    
    Args:
        tensor: Input tensor (any shape)
    
    Returns:
        Dict with shape, dtype, min, max, mean, std, p1, p50, p99
    """
    t = tensor.detach()
    if isinstance(t, torch.Tensor):
        arr = t.cpu().float().numpy()
    else:
        arr = np.asarray(t, dtype=np.float32)
    
    flat = arr.flatten()
    
    return {
        'shape': list(tensor.shape),
        'dtype': str(tensor.dtype) if isinstance(tensor, torch.Tensor) else str(arr.dtype),
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
        'mean': float(np.mean(flat)),
        'std': float(np.std(flat)),
        'p1': float(np.percentile(flat, 1)),
        'p50': float(np.percentile(flat, 50)),
        'p99': float(np.percentile(flat, 99)),
    }


# ==============================================================================
# PEAK EXTRACTION WITH NMS
# ==============================================================================

@dataclass
class Peak:
    """A detected peak in a heatmap."""
    y: int          # Row index
    x: int          # Column index
    score: float    # Peak score (0-1 after sigmoid)
    class_id: int   # Class index


def extract_nms_peaks(
    heatmap: torch.Tensor,
    k: int = 100,
    radius: int = 3,
    thresh: float = 0.1,
) -> List[Peak]:
    """
    Extract top-K peaks with radius-based NMS.
    
    This is the CORRECT way to extract peaks:
    1. Apply sigmoid to get probabilities
    2. Find local maxima (maxpool comparison)
    3. Apply threshold
    4. Take top-K per class
    5. Apply radius-based NMS to suppress nearby peaks
    
    Args:
        heatmap: (B, C, H, W) logit tensor
        k: Maximum number of peaks to return (total across all classes)
        radius: NMS radius in pixels (peaks within this distance are suppressed)
        thresh: Minimum score threshold (after sigmoid)
    
    Returns:
        List of Peak objects, sorted by score descending
    """
    # Ensure 4D
    h = heatmap.detach()
    if h.dim() == 2:
        h = h.unsqueeze(0).unsqueeze(0)
    elif h.dim() == 3:
        h = h.unsqueeze(0)
    
    B, C, H, W = h.shape
    
    # Apply sigmoid
    h_prob = torch.sigmoid(h)
    
    # Find local maxima using maxpool
    kernel_size = 2 * radius + 1
    padding = radius
    h_max = F.max_pool2d(h_prob, kernel_size, stride=1, padding=padding)
    is_peak = (h_prob == h_max) & (h_prob >= thresh)
    
    # Extract peaks
    all_peaks = []
    
    for b in range(B):
        for c in range(C):
            # Get peak locations for this class
            peak_mask = is_peak[b, c]
            if not peak_mask.any():
                continue
            
            peak_indices = torch.nonzero(peak_mask, as_tuple=False)  # (N, 2) - [y, x]
            peak_scores = h_prob[b, c, peak_indices[:, 0], peak_indices[:, 1]]
            
            for i in range(len(peak_indices)):
                all_peaks.append(Peak(
                    y=int(peak_indices[i, 0]),
                    x=int(peak_indices[i, 1]),
                    score=float(peak_scores[i]),
                    class_id=c
                ))
    
    # Sort by score descending
    all_peaks.sort(key=lambda p: p.score, reverse=True)
    
    # Apply NMS
    kept_peaks = []
    suppressed = set()
    
    for peak in all_peaks:
        if len(kept_peaks) >= k:
            break
        
        key = (peak.y, peak.x)
        if key in suppressed:
            continue
        
        kept_peaks.append(peak)
        
        # Suppress nearby peaks
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy*dy + dx*dx <= radius*radius:
                    suppressed.add((peak.y + dy, peak.x + dx))
    
    return kept_peaks


def get_gt_centers(cls_targets: torch.Tensor) -> List[Tuple[int, int, int]]:
    """
    Extract GT center locations from classification targets.
    
    Args:
        cls_targets: (B, C, H, W) target heatmap
    
    Returns:
        List of (y, x, class_id) tuples
    """
    t = cls_targets.detach()
    if t.dim() == 4:
        t = t[0]  # (C, H, W)
    
    centers = []
    C, H, W = t.shape
    
    for c in range(C):
        # Find peaks (values > 0.5 for hard_center mode)
        mask = t[c] > 0.5
        if mask.any():
            indices = torch.nonzero(mask, as_tuple=False)
            for idx in indices:
                centers.append((int(idx[0]), int(idx[1]), c))
    
    return centers


# ==============================================================================
# TENSOR EXTRACTION INTERFACE
# ==============================================================================

@dataclass
class VisualTensors:
    """
    Extracted tensors for visualization with strict semantics.
    
    Camera-only mode:
        - camera_bev: Camera BEV features (from BEVFormer)
        - lidar_bev: None (not produced)
        - fused_bev: None (not produced - DO NOT fake this!)
        - pred_heatmap: Classification logits
    
    Fusion mode:
        - camera_bev: Camera BEV features
        - lidar_bev: LiDAR BEV features (from PointPillars)
        - fused_bev: Fused BEV features (after cross-attention)
        - pred_heatmap: Classification logits
    """
    camera_bev: Optional[torch.Tensor] = None
    lidar_bev: Optional[torch.Tensor] = None
    fused_bev: Optional[torch.Tensor] = None
    pred_heatmap: Optional[torch.Tensor] = None
    mode: Literal['camera_only', 'fusion'] = 'camera_only'
    
    def available_features(self) -> List[str]:
        """Return list of available (non-None) features."""
        result = []
        if self.camera_bev is not None:
            result.append('camera_bev')
        if self.lidar_bev is not None:
            result.append('lidar_bev')
        if self.fused_bev is not None:
            result.append('fused_bev')
        return result


def extract_visual_tensors(
    preds: Dict,
    tensors: Dict,
    mode: Literal['camera_only', 'fusion']
) -> VisualTensors:
    """
    Extract tensors for visualization with strict semantic correctness.
    
    This is the ONLY function that should be used to get tensors for plotting.
    It ensures:
    - Mode-appropriate tensor selection
    - No fake "fused" features in camera-only mode
    - Explicit handling of missing tensors
    
    Args:
        preds: Model predictions dict
        tensors: Captured intermediate tensors dict
        mode: 'camera_only' or 'fusion'
    
    Returns:
        VisualTensors with properly assigned features
    """
    result = VisualTensors(mode=mode)
    
    # Extract prediction heatmap
    for key in ['cls_scores', 'cls_pred', 'heatmap', 'cls', 'output']:
        if key in preds:
            val = preds[key]
            if isinstance(val, torch.Tensor):
                result.pred_heatmap = val
                break
    
    # Extract BEV features based on mode
    if mode == 'camera_only':
        # Camera-only: only camera_bev is available
        # Look for camera BEV features
        for key in ['bev_features', 'camera_bev', 'bev_cam', 'cam_bev']:
            if key in tensors:
                val = tensors[key]
                if isinstance(val, torch.Tensor) and val.dim() == 4:
                    result.camera_bev = val
                    break
        
        # Explicitly set lidar and fused to None (not produced)
        result.lidar_bev = None
        result.fused_bev = None
        
    elif mode == 'fusion':
        # Fusion mode: all three features should be available
        
        # Camera BEV
        for key in ['camera_bev', 'bev_cam', 'cam_bev', 'camera_encoder']:
            if key in tensors:
                val = tensors[key]
                if isinstance(val, torch.Tensor) and val.dim() == 4:
                    result.camera_bev = val
                    break
        
        # LiDAR BEV
        for key in ['lidar_bev', 'bev_lidar', 'lidar_encoder', 'pillar_features']:
            if key in tensors:
                val = tensors[key]
                if isinstance(val, torch.Tensor) and val.dim() == 4:
                    result.lidar_bev = val
                    break
        
        # Fused BEV
        for key in ['fused_bev', 'bev_fused', 'fusion_out', 'fused_features']:
            if key in tensors:
                val = tensors[key]
                if isinstance(val, torch.Tensor) and val.dim() == 4:
                    result.fused_bev = val
                    break
        
        # Fallback: if no explicit fused, check for generic 'bev_features'
        if result.fused_bev is None and 'bev_features' in tensors:
            val = tensors['bev_features']
            if isinstance(val, torch.Tensor) and val.dim() == 4:
                # Only use as fused if we already have camera_bev (otherwise it's the camera output)
                if result.camera_bev is not None:
                    result.fused_bev = val
                elif result.camera_bev is None:
                    result.camera_bev = val
    
    return result


# ==============================================================================
# HEATMAP STATISTICS
# ==============================================================================

def summarize_heatmap_stats(
    heatmap: torch.Tensor,
    name: str = "heatmap"
) -> Dict:
    """
    Compute comprehensive statistics for a heatmap.
    
    Used for debugging and diagnostics.
    """
    h = heatmap.detach()
    
    if h.dim() == 4:
        h = h[0]  # Remove batch
    
    h_sigmoid = torch.sigmoid(h)
    
    # Per-class statistics
    C = h.shape[0]
    class_stats = []
    for c in range(C):
        cls_h = h_sigmoid[c]
        class_stats.append({
            'class': c,
            'max': float(cls_h.max()),
            'mean': float(cls_h.mean()),
            'above_0.5': int((cls_h > 0.5).sum()),
            'above_0.1': int((cls_h > 0.1).sum()),
        })
    
    # Overall
    return {
        'name': name,
        'shape': list(h.shape),
        'logit_range': [float(h.min()), float(h.max())],
        'prob_range': [float(h_sigmoid.min()), float(h_sigmoid.max())],
        'prob_mean': float(h_sigmoid.mean()),
        'class_stats': class_stats,
    }

