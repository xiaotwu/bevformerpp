"""
Core geometric and warping utilities for BEV features.

These functions are re-exported from modules.utils for convenient imports.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


# ==============================================================================
# LiDAR POINTS AND MASK NORMALIZATION
# ==============================================================================

def normalize_lidar_points_and_mask(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    squeeze_temporal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize LiDAR points and mask to canonical shapes for LiDAR encoder.

    This is the AUTHORITATIVE function for preparing LiDAR inputs. It handles:
    - 3D points (B, N, 4) -> pass through
    - 4D points (B, T, N, 4) -> squeeze if T==1 (when squeeze_temporal=True)
    - Mask inference when mask is None
    - Mask shape validation and normalization

    CRITICAL: Mask inference uses reduction over the LAST dimension (dim=-1),
    NOT dim=1. Using dim=1 would produce a mask of shape (B, 4) which is WRONG.

    Args:
        points: LiDAR point cloud tensor
            - Shape (B, N, 4): 3D batch of points
            - Shape (B, T, N, 4): 4D temporal batch of points
        mask: Optional boolean mask indicating valid (non-padding) points
            - Shape (B, N): for 3D points
            - Shape (B, T, N): for 4D points
            - If None, mask is inferred from non-zero points
        squeeze_temporal: If True and T==1, squeeze temporal dimension

    Returns:
        Tuple of:
            - points_3d: (B, N, 4) float tensor
            - mask_2d: (B, N) boolean tensor where True = valid point

    Raises:
        ValueError: If shapes are invalid or inconsistent

    Examples:
        >>> # 3D input
        >>> points = torch.randn(2, 1000, 4)
        >>> pts, msk = normalize_lidar_points_and_mask(points)
        >>> pts.shape, msk.shape
        (torch.Size([2, 1000, 4]), torch.Size([2, 1000]))

        >>> # 4D input with T=1
        >>> points = torch.randn(2, 1, 1000, 4)
        >>> pts, msk = normalize_lidar_points_and_mask(points)
        >>> pts.shape, msk.shape
        (torch.Size([2, 1000, 4]), torch.Size([2, 1000]))

        >>> # Invalid mask shape raises error
        >>> points = torch.randn(2, 1000, 4)
        >>> bad_mask = torch.ones(2, 4)  # Wrong shape!
        >>> normalize_lidar_points_and_mask(points, bad_mask)  # Raises ValueError
    """
    # Validate points tensor
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"points must be a torch.Tensor, got {type(points)}")

    ndim = points.dim()

    # Handle 4D temporal input (B, T, N, 4)
    if ndim == 4:
        B, T, N, C = points.shape
        if C < 4:
            raise ValueError(
                f"Points must have at least 4 features (x,y,z,intensity), got {C}. "
                f"Points shape: {points.shape}"
            )

        if squeeze_temporal and T == 1:
            # Squeeze temporal dimension for single-frame input
            points = points[:, 0]  # (B, N, 4)
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask[:, 0]  # (B, N)
                # If mask is already 2D, keep as-is
        elif T > 1 and squeeze_temporal:
            # Use last timestep for multi-frame input
            points = points[:, -1]  # (B, N, 4)
            if mask is not None:
                if mask.dim() == 3:
                    mask = mask[:, -1]  # (B, N)

    elif ndim == 3:
        B, N, C = points.shape
        if C < 4:
            raise ValueError(
                f"Points must have at least 4 features (x,y,z,intensity), got {C}. "
                f"Points shape: {points.shape}"
            )
    else:
        raise ValueError(
            f"Points must be 3D (B, N, 4) or 4D (B, T, N, 4), got {ndim}D tensor "
            f"with shape {points.shape}"
        )

    # At this point, points should be 3D (B, N, C)
    if points.dim() != 3:
        raise ValueError(
            f"After normalization, points should be 3D, got {points.dim()}D. "
            f"This is a bug in normalize_lidar_points_and_mask."
        )

    B, N, C = points.shape

    # Normalize mask
    if mask is None:
        # Infer mask from non-zero coordinates
        # CRITICAL: Use dim=-1 (last dimension) to reduce over x,y,z
        # This produces (B, N) shape, NOT (B, 4)!
        xyz = points[..., :3]  # (B, N, 3)
        mask = (xyz.abs().sum(dim=-1) > 0)  # (B, N) - sum over x,y,z features
    else:
        # Validate and normalize provided mask
        mask = mask.to(points.device)

        if mask.dim() == 3:
            # (B, T, N) -> squeeze if T matches
            if mask.shape[1] == 1:
                mask = mask[:, 0]  # (B, N)
            else:
                # Use last timestep
                mask = mask[:, -1]  # (B, N)

        if mask.dim() != 2:
            raise ValueError(
                f"Mask must be 2D (B, N) after normalization, got {mask.dim()}D "
                f"with shape {mask.shape}. "
                f"If you have a 3D mask (B, T, N), ensure T dimension is handled."
            )

        # Validate mask shape matches points
        if mask.shape != (B, N):
            raise ValueError(
                f"Mask shape {mask.shape} doesn't match points (B={B}, N={N}). "
                f"Expected mask shape: ({B}, {N}). "
                f"Common errors:\n"
                f"  - (B, 4) means you used .any(dim=1) instead of .any(dim=-1)\n"
                f"  - (B, N, 4) means you forgot to reduce over the feature dimension\n"
                f"  - (B, T, N) means temporal dimension wasn't squeezed"
            )

        # Convert to boolean
        if mask.dtype != torch.bool:
            mask = mask > 0.5

    # Ensure mask is boolean
    mask = mask.bool()

    # Keep only first 4 features
    points = points[..., :4].contiguous()

    return points, mask


def validate_lidar_mask_shape(
    mask: torch.Tensor,
    expected_batch: int,
    expected_points: int,
    context: str = ""
) -> None:
    """
    Validate that a LiDAR mask has the correct shape.

    This function provides clear error messages for common mask shape bugs.

    Args:
        mask: The mask tensor to validate
        expected_batch: Expected batch size (B)
        expected_points: Expected number of points (N)
        context: Optional context string for error messages

    Raises:
        ValueError: If mask shape is invalid
    """
    ctx = f" [{context}]" if context else ""

    if mask.dim() != 2:
        raise ValueError(
            f"LiDAR mask{ctx} must be 2D (B, N), got {mask.dim()}D "
            f"with shape {mask.shape}. "
            f"Expected shape: ({expected_batch}, {expected_points})"
        )

    if mask.shape[0] != expected_batch:
        raise ValueError(
            f"LiDAR mask{ctx} batch size mismatch: got {mask.shape[0]}, "
            f"expected {expected_batch}. Mask shape: {mask.shape}"
        )

    if mask.shape[1] != expected_points:
        raise ValueError(
            f"LiDAR mask{ctx} points dimension mismatch: "
            f"got {mask.shape[1]}, expected {expected_points}. "
            f"Mask shape: {mask.shape}. "
            f"Common errors:\n"
            f"  - If mask has {mask.shape[1]}=4, you likely used .any(dim=1) "
            f"instead of .any(dim=-1) when inferring mask from points\n"
            f"  - If mask shape doesn't match points, check that mask and points "
            f"come from the same batch"
        )


def align_bev_features(features: torch.Tensor, ego_transform: torch.Tensor,
                       bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)) -> torch.Tensor:
    """
    Warp BEV features from previous frame to current frame using ego-motion.
    Uses bilinear interpolation for smooth warping.

    Args:
        features: BEV features from previous frame, shape (B, C, H, W)
        ego_transform: SE(3) transformation matrix from previous to current frame,
                      shape (B, 4, 4)
        bev_range: BEV range (x_min, x_max, y_min, y_max) in meters

    Returns:
        Warped BEV features aligned to current frame, shape (B, C, H, W)
    """
    B, C, H, W = features.shape
    device = features.device

    try:
        # Invert transformation: we need T_{curr->prev} for sampling
        ego_transform_inv = torch.inverse(ego_transform)
    except RuntimeError:
        return features

    rotation_2d = ego_transform_inv[:, :2, :2]  # (B, 2, 2)
    translation_2d = ego_transform_inv[:, :2, 3]  # (B, 2)

    x_min, x_max, y_min, y_max = bev_range
    x_range = x_max - x_min
    y_range = y_max - y_min

    translation_normalized = torch.zeros(B, 2, device=device, dtype=features.dtype)
    translation_normalized[:, 0] = translation_2d[:, 0] * 2.0 / x_range
    translation_normalized[:, 1] = translation_2d[:, 1] * 2.0 / y_range

    affine_matrix = torch.zeros(B, 2, 3, device=device, dtype=features.dtype)
    affine_matrix[:, :2, :2] = rotation_2d
    affine_matrix[:, :, 2] = translation_normalized

    grid = F.affine_grid(affine_matrix, [B, C, H, W], align_corners=False)

    warped_features = F.grid_sample(
        features,
        grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    return warped_features


def generate_grid_from_transform(ego_transform: torch.Tensor, H: int, W: int,
                                  bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)) -> torch.Tensor:
    """
    Generate sampling grid from ego-motion transformation.

    Args:
        ego_transform: SE(3) transformation matrix, shape (B, 4, 4)
        H: Height of BEV grid
        W: Width of BEV grid
        bev_range: BEV range (x_min, x_max, y_min, y_max) in meters

    Returns:
        Sampling grid of shape (B, H, W, 2)
    """
    B = ego_transform.shape[0]
    device = ego_transform.device
    dtype = ego_transform.dtype

    ego_transform_inv = torch.inverse(ego_transform)
    rotation_2d = ego_transform_inv[:, :2, :2]
    translation_2d = ego_transform_inv[:, :2, 3]

    x_min, x_max, y_min, y_max = bev_range
    x_range = x_max - x_min
    y_range = y_max - y_min

    translation_normalized = torch.zeros(B, 2, device=device, dtype=dtype)
    translation_normalized[:, 0] = translation_2d[:, 0] * 2.0 / x_range
    translation_normalized[:, 1] = translation_2d[:, 1] * 2.0 / y_range

    affine_matrix = torch.zeros(B, 2, 3, device=device, dtype=dtype)
    affine_matrix[:, :2, :2] = rotation_2d
    affine_matrix[:, :, 2] = translation_normalized

    grid = F.affine_grid(affine_matrix, [B, 1, H, W], align_corners=False)
    return grid


def compute_visibility_mask(ego_transform: torch.Tensor, H: int, W: int,
                            bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)) -> torch.Tensor:
    """
    Compute visibility mask for warped BEV features.
    Marks pixels that are out of bounds as invalid.

    Args:
        ego_transform: SE(3) transformation matrix, shape (B, 4, 4)
        H: Height of BEV grid
        W: Width of BEV grid
        bev_range: Tuple of (x_min, x_max, y_min, y_max) in meters

    Returns:
        Visibility mask of shape (B, 1, H, W) with values in [0, 1]
    """
    grid = generate_grid_from_transform(ego_transform, H, W, bev_range=bev_range)
    valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
    valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
    valid_mask = valid_x & valid_y
    return valid_mask.float().unsqueeze(1)


def warp_bev(prev_bev: torch.Tensor, ego_motion: torch.Tensor,
             bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2)) -> torch.Tensor:
    """
    Legacy wrapper for align_bev_features.
    """
    return align_bev_features(prev_bev, ego_motion, bev_range=bev_range)

