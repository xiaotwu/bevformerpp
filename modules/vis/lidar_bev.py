"""
LiDAR BEV projection and density visualization utilities.

This module provides robust LiDAR point cloud to BEV projection with:
- Proper coordinate handling and axis conventions
- Diagnostic metrics for debugging alignment issues
- Scatter overlay sanity checks
- Robust extraction for various input formats (B,T,N,4), (B,N,4), (N,4), etc.

Coordinate Convention:
    - BEV image: row index = y, column index = x
    - Origin: bottom-left when using origin='lower' in imshow
    - X axis: right (longitudinal, vehicle forward direction in nuScenes)
    - Y axis: up in image (lateral, left direction in nuScenes)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# LIDAR POINTS EXTRACTION AND VALIDATION
# ==============================================================================

def validate_points_np(
    points_np: np.ndarray,
    require_cols: int = 3,
    context: str = ""
) -> np.ndarray:
    """
    Validate that points array has the expected shape for visualization.

    Args:
        points_np: Numpy array to validate
        require_cols: Minimum number of columns required (default: 3 for xyz)
        context: Optional context string for error messages

    Returns:
        Validated points array (may be a view or copy)

    Raises:
        ValueError: If shape is invalid with helpful error message

    Example:
        >>> pts = np.random.randn(100, 4)
        >>> pts = validate_points_np(pts, require_cols=3)
        >>> pts.shape
        (100, 4)
    """
    ctx = f" [{context}]" if context else ""

    # Check dtype
    if points_np.dtype == object:
        # Try to recover from object array
        try:
            points_np = np.stack(points_np.flatten(), axis=0)
            logger.warning(
                f"validate_points_np{ctx}: Recovered object array by stacking. "
                f"New shape: {points_np.shape}"
            )
        except Exception as e:
            sample = points_np.flatten()[:3] if points_np.size > 0 else []
            raise ValueError(
                f"LiDAR points{ctx} has dtype=object which is invalid. "
                f"This usually indicates incorrect slicing or unpacking. "
                f"Sample elements: {sample}. Error: {e}"
            )

    # Check ndim
    if points_np.ndim != 2:
        raise ValueError(
            f"LiDAR points{ctx} must be 2D (N, C), got {points_np.ndim}D "
            f"with shape {points_np.shape}. "
            f"If you have batched data (B,N,C) or temporal data (B,T,N,C), "
            f"use extract_lidar_points_np() to extract a single frame."
        )

    N, C = points_np.shape

    # Check for transposed array
    if C > N and N in {3, 4}:
        logger.warning(
            f"validate_points_np{ctx}: Points appear transposed "
            f"(shape {points_np.shape}). Transposing to (N, C)."
        )
        points_np = points_np.T
        N, C = points_np.shape

    # Check columns
    if C < require_cols:
        sample = points_np[:3].tolist() if N > 0 else []
        raise ValueError(
            f"LiDAR points{ctx} must have at least {require_cols} columns "
            f"(x, y{'z' if require_cols >= 3 else ''}), got {C}. "
            f"Shape: {points_np.shape}, dtype: {points_np.dtype}. "
            f"Sample (first 3 rows): {sample}. "
            f"If C=1, this usually means an object array was incorrectly "
            f"indexed (e.g., array of arrays), or the data format is wrong."
        )

    # Ensure float32
    if points_np.dtype != np.float32:
        points_np = points_np.astype(np.float32)

    return points_np


def extract_lidar_points_np(
    lidar_points: Any,
    lidar_mask: Optional[Any] = None,
    *,
    take_t: str = "last",
    take_b: int = 0,
    require_cols: int = 3,
) -> np.ndarray:
    """
    Extract LiDAR points as a clean numpy array (N, 4) or (N, 3).

    This is the AUTHORITATIVE function for preparing LiDAR points for visualization.
    It handles all common input formats and edge cases robustly.

    Supported input formats:
        - torch.Tensor: (B, T, N, 4), (B, N, 4), (T, N, 4), (N, 4)
        - np.ndarray: same shapes as above
        - dict: {"lidar_points": ..., "lidar_mask": ...} or {"points": ...}
        - tuple/list: (points, mask)

    Args:
        lidar_points: LiDAR point cloud in various formats
        lidar_mask: Optional mask (same leading dims as points without last coord dim)
            True = valid point, False = padding
        take_t: Which timestep to use if temporal: "last" or "first"
        take_b: Which batch sample to use (default: 0)
        require_cols: Minimum columns required (default: 3 for xyz)

    Returns:
        float32 numpy array of shape (N, 4) or (N, 3), with padding filtered if mask provided

    Raises:
        ValueError: If input cannot be converted to valid points array

    Example:
        >>> # 4D temporal input
        >>> pts = torch.randn(1, 1, 35000, 4)
        >>> pts_np = extract_lidar_points_np(pts)
        >>> pts_np.shape
        (35000, 4)

        >>> # With mask
        >>> mask = torch.zeros(1, 1, 35000, dtype=torch.bool)
        >>> mask[:, :, :100] = True
        >>> pts_np = extract_lidar_points_np(pts, mask)
        >>> pts_np.shape
        (100, 4)
    """
    # Handle dict input
    if isinstance(lidar_points, dict):
        # Try common key names
        for key in ['lidar_points', 'points', 'lidar']:
            if key in lidar_points:
                points_data = lidar_points[key]
                break
        else:
            raise ValueError(
                f"Dict input must contain 'lidar_points' or 'points' key. "
                f"Available keys: {list(lidar_points.keys())}"
            )

        # Check for mask in dict if not provided
        if lidar_mask is None:
            for key in ['lidar_mask', 'mask', 'lidar_mask_seq']:
                if key in lidar_points:
                    lidar_mask = lidar_points[key]
                    break

        lidar_points = points_data

    # Handle tuple/list input
    if isinstance(lidar_points, (tuple, list)):
        if len(lidar_points) >= 2 and lidar_mask is None:
            lidar_points, lidar_mask = lidar_points[0], lidar_points[1]
        elif len(lidar_points) == 1:
            lidar_points = lidar_points[0]
        else:
            raise ValueError(
                f"Tuple/list input expected (points,) or (points, mask), "
                f"got length {len(lidar_points)}"
            )

    # Convert torch to numpy
    if isinstance(lidar_points, torch.Tensor):
        lidar_points = lidar_points.detach().cpu()
        if not lidar_points.is_contiguous():
            lidar_points = lidar_points.contiguous()
        points_np = lidar_points.numpy()
    elif isinstance(lidar_points, np.ndarray):
        points_np = lidar_points
    else:
        raise ValueError(
            f"Unsupported lidar_points type: {type(lidar_points)}. "
            f"Expected torch.Tensor, np.ndarray, dict, or tuple."
        )

    # Handle object dtype (array of arrays)
    if points_np.dtype == object:
        try:
            # Flatten and stack
            flat = points_np.flatten()
            points_np = np.stack([np.asarray(x) for x in flat], axis=0)
            logger.warning(
                f"extract_lidar_points_np: Recovered object array by stacking. "
                f"New shape: {points_np.shape}"
            )
        except Exception as e:
            raise ValueError(
                f"Failed to convert object array to numeric array. "
                f"Shape: {points_np.shape}, sample: {points_np.flatten()[:3]}. "
                f"Error: {e}"
            )

    # Normalize dimensions: reduce to (N, C)
    ndim = points_np.ndim

    if ndim == 4:
        # (B, T, N, C)
        B, T, N, C = points_np.shape
        b_idx = min(take_b, B - 1)
        t_idx = T - 1 if take_t == "last" else 0
        points_np = points_np[b_idx, t_idx]  # (N, C)

    elif ndim == 3:
        # Could be (B, N, C) or (T, N, C)
        dim0, dim1, dim2 = points_np.shape

        if dim2 in {3, 4} and dim1 > 16:
            # Likely (B, N, C) or (T, N, C) where N is large
            if dim0 <= 16:
                # Likely temporal (T, N, C) - small first dim
                t_idx = dim0 - 1 if take_t == "last" else 0
                points_np = points_np[t_idx]  # (N, C)
            else:
                # Likely batch (B, N, C)
                b_idx = min(take_b, dim0 - 1)
                points_np = points_np[b_idx]  # (N, C)
        elif dim1 in {3, 4} and dim0 > dim1:
            # (N, C, ??) - unusual but handle
            points_np = points_np[:, :, 0]  # Take first slice
        else:
            # Default: treat as (B, N, C)
            b_idx = min(take_b, dim0 - 1)
            points_np = points_np[b_idx]  # (N, C)

    elif ndim == 2:
        # (N, C) - already correct, or (C, N) if transposed
        pass  # Will be validated below

    elif ndim == 1:
        raise ValueError(
            f"LiDAR points is 1D with shape {points_np.shape}. "
            f"Expected at least 2D (N, C)."
        )

    elif ndim == 0:
        raise ValueError("LiDAR points is a scalar, expected array.")

    # Validate final shape
    points_np = validate_points_np(points_np, require_cols=require_cols, context="after extraction")

    # Process mask if provided
    if lidar_mask is not None:
        mask_np = _extract_mask_np(lidar_mask, points_np.shape[0], take_t=take_t, take_b=take_b)
        if mask_np is not None:
            points_np = points_np[mask_np]

    return points_np


def _extract_mask_np(
    mask: Any,
    num_points: int,
    *,
    take_t: str = "last",
    take_b: int = 0,
) -> Optional[np.ndarray]:
    """
    Extract mask as 1D boolean numpy array.

    Args:
        mask: Mask in various formats
        num_points: Expected number of points (N)
        take_t: Which timestep to use if temporal
        take_b: Which batch sample to use

    Returns:
        1D boolean numpy array of length N, or None if mask is invalid
    """
    if mask is None:
        return None

    # Convert torch to numpy
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    elif not isinstance(mask, np.ndarray):
        try:
            mask = np.asarray(mask)
        except Exception:
            logger.warning(f"Could not convert mask of type {type(mask)} to numpy")
            return None

    # Normalize dimensions
    ndim = mask.ndim

    if ndim == 3:
        # (B, T, N)
        B, T, N = mask.shape
        b_idx = min(take_b, B - 1)
        t_idx = T - 1 if take_t == "last" else 0
        mask = mask[b_idx, t_idx]  # (N,)

    elif ndim == 2:
        # (B, N) or (T, N)
        dim0, dim1 = mask.shape
        if dim1 == num_points:
            # Likely (B, N) or (T, N)
            idx = dim0 - 1 if take_t == "last" else 0
            idx = min(take_b if dim0 > 1 else idx, dim0 - 1)
            mask = mask[idx]  # (N,)
        elif dim0 == num_points:
            # Transposed (N, T) - unusual
            t_idx = dim1 - 1 if take_t == "last" else 0
            mask = mask[:, t_idx]  # (N,)
        else:
            logger.warning(
                f"Mask shape {mask.shape} doesn't match num_points={num_points}"
            )
            return None

    elif ndim == 1:
        # (N,) - already correct
        if len(mask) != num_points:
            logger.warning(
                f"Mask length {len(mask)} doesn't match num_points={num_points}"
            )
            return None

    else:
        logger.warning(f"Unsupported mask ndim: {ndim}")
        return None

    # Convert to boolean
    if mask.dtype != bool:
        mask = mask > 0.5

    return mask


def project_lidar_to_bev(
    points_xyz: Union[np.ndarray, torch.Tensor, Any],
    x_range: Tuple[float, float] = (-51.2, 51.2),
    y_range: Tuple[float, float] = (-51.2, 51.2),
    resolution: float = 0.512,
    z_range: Optional[Tuple[float, float]] = (-5.0, 3.0),
    lidar_mask: Optional[Any] = None,
) -> np.ndarray:
    """
    Project LiDAR points to BEV density map.

    This function now uses extract_lidar_points_np() for robust input handling,
    supporting various tensor shapes and formats.

    Args:
        points_xyz: Point cloud of shape (N, 3+), (B, N, 4), (B, T, N, 4), etc.
                   Can be numpy array, torch.Tensor, dict, or tuple.
        x_range: (x_min, x_max) range in meters
        y_range: (y_min, y_max) range in meters
        resolution: Grid resolution in meters per pixel
        z_range: Optional (z_min, z_max) filter. None = no z filtering.
        lidar_mask: Optional mask to filter padding points.

    Returns:
        density: (H, W) density map where H = (y_max-y_min)/resolution,
                 W = (x_max-x_min)/resolution

    Coordinate convention:
        - density[row, col] corresponds to y, x
        - row 0 = y_min (bottom when origin='lower')
        - col 0 = x_min (left side)
    """
    # Use robust extraction
    try:
        points_np = extract_lidar_points_np(
            points_xyz, lidar_mask=lidar_mask, require_cols=3
        )
    except ValueError as e:
        logger.error(f"project_lidar_to_bev: Invalid input - {e}")
        # Return empty density map
        W = int(np.ceil((x_range[1] - x_range[0]) / resolution))
        H = int(np.ceil((y_range[1] - y_range[0]) / resolution))
        return np.zeros((H, W), dtype=np.float32)

    if len(points_np) == 0:
        W = int(np.ceil((x_range[1] - x_range[0]) / resolution))
        H = int(np.ceil((y_range[1] - y_range[0]) / resolution))
        return np.zeros((H, W), dtype=np.float32)

    x = points_np[:, 0]
    y = points_np[:, 1]
    z = points_np[:, 2] if points_np.shape[1] >= 3 else np.zeros_like(x)

    # Filter out zero-padded points (common in batched data)
    # A point is considered padding if ALL of x, y, z are exactly 0
    valid_mask = ~((x == 0) & (y == 0) & (z == 0))

    # Apply z range filter if specified
    if z_range is not None:
        z_mask = (z >= z_range[0]) & (z <= z_range[1])
        valid_mask = valid_mask & z_mask

    # Apply x/y range filter
    xy_mask = (x >= x_range[0]) & (x < x_range[1]) & (y >= y_range[0]) & (y < y_range[1])
    valid_mask = valid_mask & xy_mask

    # Get valid points
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]

    # Compute grid dimensions
    W = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    H = int(np.ceil((y_range[1] - y_range[0]) / resolution))

    if len(x_valid) == 0:
        return np.zeros((H, W), dtype=np.float32)

    # Create 2D histogram
    # histogram2d returns (H, W) where first axis is x, second is y
    # We want row=y, col=x, so we pass (y, x) and the result is (H_y, W_x)
    density, _, _ = np.histogram2d(
        y_valid, x_valid,
        bins=[H, W],
        range=[[y_range[0], y_range[1]], [x_range[0], x_range[1]]]
    )

    return density.astype(np.float32)


def debug_lidar_bev(
    points_xyz: Union[np.ndarray, torch.Tensor, Any],
    x_range: Tuple[float, float] = (-51.2, 51.2),
    y_range: Tuple[float, float] = (-51.2, 51.2),
    z_range: Optional[Tuple[float, float]] = (-5.0, 3.0),
    resolution: float = 0.512,
    lidar_mask: Optional[Any] = None,
) -> Dict:
    """
    Compute diagnostic metrics for LiDAR BEV projection.

    This function now uses extract_lidar_points_np() for robust input handling.

    Args:
        points_xyz: Point cloud (N, 3+), (B, N, 4), (B, T, N, 4), etc.
        x_range, y_range, z_range: BEV extent and z filter
        resolution: Grid resolution
        lidar_mask: Optional mask to filter padding points.

    Returns:
        Dictionary with diagnostic info:
            - num_points_total: Total points (including padding)
            - num_points_valid: Non-padding points
            - num_points_in_range: Points after x/y/z filtering
            - fraction_clipped: Fraction of valid points outside range
            - xyz_min: [x_min, y_min, z_min] of valid points
            - xyz_max: [x_max, y_max, z_max] of valid points
            - density_sum: Total count in density map
            - density_max: Maximum density in a single cell
            - density_nonzero_fraction: Fraction of cells with >0 points
    """
    # Use robust extraction
    try:
        points_np = extract_lidar_points_np(
            points_xyz, lidar_mask=lidar_mask, require_cols=2
        )
    except ValueError as e:
        logger.error(f"debug_lidar_bev: Invalid input - {e}")
        W = int(np.ceil((x_range[1] - x_range[0]) / resolution))
        H = int(np.ceil((y_range[1] - y_range[0]) / resolution))
        return {
            'num_points_total': 0,
            'num_points_valid': 0,
            'num_points_in_range': 0,
            'fraction_clipped': 0.0,
            'xyz_min': [0.0, 0.0, 0.0],
            'xyz_max': [0.0, 0.0, 0.0],
            'density_sum': 0.0,
            'density_max': 0.0,
            'density_nonzero_fraction': 0.0,
            'error': str(e),
        }

    if len(points_np) == 0:
        W = int(np.ceil((x_range[1] - x_range[0]) / resolution))
        H = int(np.ceil((y_range[1] - y_range[0]) / resolution))
        return {
            'num_points_total': 0,
            'num_points_valid': 0,
            'num_points_in_range': 0,
            'fraction_clipped': 0.0,
            'xyz_min': [0.0, 0.0, 0.0],
            'xyz_max': [0.0, 0.0, 0.0],
            'density_sum': 0.0,
            'density_max': 0.0,
            'density_nonzero_fraction': 0.0,
        }

    x = points_np[:, 0]
    y = points_np[:, 1]
    z = points_np[:, 2] if points_np.shape[1] >= 3 else np.zeros_like(x)

    num_total = len(x)

    # Valid (non-padding) points
    valid_mask = ~((x == 0) & (y == 0) & (z == 0))
    num_valid = valid_mask.sum()

    x_v, y_v, z_v = x[valid_mask], y[valid_mask], z[valid_mask]

    # In-range points
    in_range_mask = (
        (x_v >= x_range[0]) & (x_v < x_range[1]) &
        (y_v >= y_range[0]) & (y_v < y_range[1])
    )
    if z_range is not None:
        in_range_mask = in_range_mask & (z_v >= z_range[0]) & (z_v <= z_range[1])

    num_in_range = in_range_mask.sum()

    # Compute density (reuse extracted points)
    density = project_lidar_to_bev(
        points_np, x_range, y_range, resolution, z_range
    )

    result = {
        'num_points_total': int(num_total),
        'num_points_valid': int(num_valid),
        'num_points_in_range': int(num_in_range),
        'fraction_clipped': 1.0 - (num_in_range / max(num_valid, 1)),
        'xyz_min': [float(x_v.min()) if len(x_v) > 0 else 0.0,
                    float(y_v.min()) if len(y_v) > 0 else 0.0,
                    float(z_v.min()) if len(z_v) > 0 else 0.0],
        'xyz_max': [float(x_v.max()) if len(x_v) > 0 else 0.0,
                    float(y_v.max()) if len(y_v) > 0 else 0.0,
                    float(z_v.max()) if len(z_v) > 0 else 0.0],
        'density_sum': float(density.sum()),
        'density_max': float(density.max()),
        'density_nonzero_fraction': float((density > 0).sum() / max(density.size, 1)),
    }

    return result


def plot_lidar_bev_density(
    points_xyz: Union[np.ndarray, torch.Tensor],
    ax: Optional[plt.Axes] = None,
    x_range: Tuple[float, float] = (-51.2, 51.2),
    y_range: Tuple[float, float] = (-51.2, 51.2),
    z_range: Optional[Tuple[float, float]] = (-5.0, 3.0),
    resolution: float = 0.512,
    title: str = "LiDAR BEV Density",
    cmap: str = "viridis",
    log_scale: bool = True,
    show_scatter: bool = False,
    scatter_downsample: int = 100,
) -> Tuple[plt.Axes, Dict]:
    """
    Plot LiDAR points as BEV density map with diagnostics.
    
    Args:
        points_xyz: Point cloud (N, 3+) or (B, N, 3+)
        ax: Matplotlib axes (created if None)
        x_range, y_range, z_range: BEV extent
        resolution: Grid resolution
        title: Plot title
        cmap: Colormap
        log_scale: Use log(1+density) for better visualization
        show_scatter: Overlay downsampled point scatter
        scatter_downsample: Downsample factor for scatter overlay
    
    Returns:
        ax: Matplotlib axes
        diag: Diagnostic dictionary from debug_lidar_bev
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Handle batched input - take first sample
    if isinstance(points_xyz, torch.Tensor):
        if points_xyz.dim() == 3:  # (B, N, C)
            points_xyz = points_xyz[0]
        points_np = points_xyz.cpu().numpy()
    elif isinstance(points_xyz, np.ndarray):
        if points_xyz.ndim == 3:  # (B, N, C)
            points_xyz = points_xyz[0]
        points_np = points_xyz
    else:
        ax.text(0.5, 0.5, 'Invalid LiDAR data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax, {}
    
    if points_np is None or len(points_np) == 0:
        ax.text(0.5, 0.5, 'LiDAR N/A', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return ax, {}
    
    # Get diagnostics
    diag = debug_lidar_bev(points_np, x_range, y_range, z_range, resolution)
    
    # Compute density
    density = project_lidar_to_bev(points_np, x_range, y_range, resolution, z_range)
    
    # Apply log scale for better visualization
    if log_scale:
        density_vis = np.log1p(density)
    else:
        density_vis = density
    
    # Plot density
    extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
    im = ax.imshow(density_vis, origin='lower', extent=extent, cmap=cmap, aspect='equal')
    
    # Optional scatter overlay for sanity check
    if show_scatter:
        # Get in-range points
        x = points_np[:, 0]
        y = points_np[:, 1]
        z = points_np[:, 2] if points_np.shape[1] >= 3 else np.zeros_like(x)
        
        valid = ~((x == 0) & (y == 0) & (z == 0))
        in_range = (
            valid &
            (x >= x_range[0]) & (x < x_range[1]) &
            (y >= y_range[0]) & (y < y_range[1])
        )
        if z_range is not None:
            in_range = in_range & (z >= z_range[0]) & (z <= z_range[1])
        
        x_in = x[in_range]
        y_in = y[in_range]
        
        # Downsample for visibility
        if len(x_in) > scatter_downsample:
            idx = np.random.choice(len(x_in), scatter_downsample, replace=False)
            x_in, y_in = x_in[idx], y_in[idx]
        
        ax.scatter(x_in, y_in, c='red', s=1, alpha=0.5, marker='.')
    
    # Add diagnostics to title
    title_full = f"{title}\n(valid={diag['num_points_in_range']}, clip={diag['fraction_clipped']:.1%})"
    ax.set_title(title_full, fontsize=10)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, label='log(1+count)' if log_scale else 'count')
    
    return ax, diag



