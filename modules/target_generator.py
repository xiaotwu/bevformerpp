"""
Ground Truth Target Generation for BEV Object Detection.

This module implements target generation following the CenterNet/CenterPoint methodology:
- Gaussian heatmaps for object centers (classification targets)
- Dense regression targets for bounding box parameters
- Binary masks for valid regression locations

References:
    - CenterNet: Objects as Points (Zhou et al., 2019)
    - CenterPoint: Center-based 3D Object Detection (Yin et al., 2021)
    - CornerNet: Detecting Objects as Paired Keypoints (Law & Deng, 2018)
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .data_structures import Box3D, BEVGridConfig

# Module logger
logger = logging.getLogger(__name__)


# nuScenes class mapping (10 classes as per configuration)
NUSCENES_CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier'
]

# Class name to index mapping
CLASS_TO_IDX = {name: idx for idx, name in enumerate(NUSCENES_CLASSES)}


@dataclass
class TargetConfig:
    """Configuration for target generation."""
    num_classes: int = 10
    bev_h: int = 200
    bev_w: int = 200
    x_range: Tuple[float, float] = (-51.2, 51.2)
    y_range: Tuple[float, float] = (-51.2, 51.2)
    gaussian_overlap: float = 0.01  # IoU threshold for Gaussian radius (reduced to enforce sharper peaks)
    min_radius: int = 1  # Minimum Gaussian radius in pixels (reduced to enforce sharper peaks)
    max_objs: int = 500  # Maximum objects per sample
    heatmap_mode: str = "gaussian"  # Options: "gaussian", "hard_center", "hard_center_radius1"


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.1) -> float:
    """
    Compute Gaussian radius based on object size and desired overlap.

    Following CenterNet, we compute the radius such that a pair of Gaussians
    placed at corner positions would have the specified IoU overlap.

    Args:
        det_size: (height, width) of the object in pixels
        min_overlap: Minimum IoU overlap threshold

    Returns:
        Gaussian radius

    Reference:
        CornerNet: Detecting Objects as Paired Keypoints (Law & Deng, 2018)
    """
    height, width = det_size

    # Three cases based on object aspect ratio
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap: np.ndarray, center: Tuple[int, int], radius: int,
                  k: float = 1.0) -> np.ndarray:
    """
    Draw a 2D Gaussian on the heatmap at the specified center.

    Uses element-wise maximum to handle overlapping Gaussians,
    ensuring that closer objects maintain higher peaks.

    Args:
        heatmap: 2D array to draw on, shape (H, W)
        center: (x, y) center coordinates in pixels
        radius: Gaussian radius
        k: Peak value (default: 1.0)

    Returns:
        Updated heatmap
    """
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape

    # Compute valid region
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    if left + right <= 0 or top + bottom <= 0:
        return heatmap

    # Extract regions
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    # Use maximum to preserve peaks of closer objects
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def _gaussian_2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.

    Args:
        shape: (height, width) of the kernel
        sigma: Standard deviation

    Returns:
        2D Gaussian kernel normalized to [0, 1]
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class TargetGenerator:
    """
    Generates training targets from 3D bounding box annotations.

    This class converts Box3D annotations to:
    1. Classification heatmaps: Gaussian peaks at object centers
    2. Regression targets: Per-pixel bounding box parameters
    3. Regression masks: Binary masks indicating valid regression locations

    The methodology follows CenterNet/CenterPoint with adaptations for BEV space.
    """

    def __init__(self, config: Optional[TargetConfig] = None):
        """
        Initialize target generator.

        Args:
            config: Target generation configuration
        """
        self.config = config or TargetConfig()

        # Compute resolution
        self.x_res = (self.config.x_range[1] - self.config.x_range[0]) / self.config.bev_w
        self.y_res = (self.config.y_range[1] - self.config.y_range[0]) / self.config.bev_h

        # Log heatmap mode once at DEBUG level
        logger.debug(f"TargetGenerator initialized with heatmap_mode={self.config.heatmap_mode}")

    def generate(self, annotations: List[Box3D]) -> Dict[str, torch.Tensor]:
        """
        Generate training targets from annotations.

        Args:
            annotations: List of Box3D ground truth annotations

        Returns:
            Dictionary containing:
                - 'cls_targets': (num_classes, H, W) classification heatmaps
                - 'bbox_targets': (7, H, W) regression targets [dx, dy, z, log(w), log(l), log(h), sin(yaw), cos(yaw)]
                - 'reg_mask': (1, H, W) binary mask for valid regression locations
                - 'indices': (max_objs, 2) center indices for each object
                - 'num_objs': Number of valid objects
        """
        cfg = self.config
        logger.debug(f"Generating targets: heatmap_mode={cfg.heatmap_mode}, n_annotations={len(annotations)}")

        # Initialize targets
        cls_targets = np.zeros((cfg.num_classes, cfg.bev_h, cfg.bev_w), dtype=np.float32)
        bbox_targets = np.zeros((7, cfg.bev_h, cfg.bev_w), dtype=np.float32)
        reg_mask = np.zeros((1, cfg.bev_h, cfg.bev_w), dtype=np.float32)

        # Object indices for loss computation
        indices = np.zeros((cfg.max_objs, 2), dtype=np.int64)

        num_objs = 0

        for ann in annotations:
            # Get class index
            cls_idx = self._get_class_index(ann.label)
            if cls_idx is None:
                continue  # Skip unknown classes

            # Convert center from meters to pixels
            cx_px, cy_px = self._meters_to_pixels(ann.center[0], ann.center[1])

            # Skip objects outside BEV bounds
            if not self._is_in_bounds(cx_px, cy_px):
                continue

            # Convert size from meters to pixels (for Gaussian radius)
            w_px = ann.size[0] / self.x_res
            l_px = ann.size[1] / self.y_res

            # Compute Gaussian radius (used only in gaussian mode)
            radius = max(
                int(gaussian_radius((l_px, w_px), cfg.gaussian_overlap)),
                cfg.min_radius
            )

            # Quantize center to integer pixel
            cx_int, cy_int = int(cx_px), int(cy_px)
            
            # Clamp indices to valid bounds (safety check)
            cx_int = max(0, min(cx_int, cfg.bev_w - 1))
            cy_int = max(0, min(cy_int, cfg.bev_h - 1))

            # Compute regression targets at center location
            # Offset from quantized center (sub-pixel accuracy)
            dx = cx_px - cx_int
            dy = cy_px - cy_int

            # Height (z-coordinate)
            z = ann.center[2]

            # Log-encoded size (numerical stability)
            log_w = np.log(max(ann.size[0], 1e-4))
            log_l = np.log(max(ann.size[1], 1e-4))
            log_h = np.log(max(ann.size[2], 1e-4))

            # Yaw angle (sin/cos encoding for continuity)
            # Note: We use raw yaw here; sin/cos encoding is optional
            yaw = ann.yaw

            # Store regression targets
            bbox_targets[0, cy_int, cx_int] = dx  # Sub-pixel x offset
            bbox_targets[1, cy_int, cx_int] = dy  # Sub-pixel y offset
            bbox_targets[2, cy_int, cx_int] = z   # Height
            bbox_targets[3, cy_int, cx_int] = log_w  # Log width
            bbox_targets[4, cy_int, cx_int] = log_l  # Log length
            bbox_targets[5, cy_int, cx_int] = log_h  # Log height
            bbox_targets[6, cy_int, cx_int] = yaw    # Yaw angle

            # Mark this location as valid for regression
            reg_mask[0, cy_int, cx_int] = 1.0

            # Store object index
            if num_objs < cfg.max_objs:
                indices[num_objs] = [cy_int, cx_int]
                num_objs += 1

            # Generate heatmap target based on mode
            # NOTE: hard_center must NOT call draw_gaussian
            if cfg.heatmap_mode == "hard_center":
                # Hard center: ONLY set center cell to 1.0, NO gaussian
                cls_targets[cls_idx, cy_int, cx_int] = 1.0
                # Skip to next object - no other heatmap writes
                continue
            elif cfg.heatmap_mode == "gaussian":
                # Original Gaussian heatmap behavior
                draw_gaussian(cls_targets[cls_idx], (cx_int, cy_int), radius)
            elif cfg.heatmap_mode == "hard_center_radius1":
                # Hard center with tiny halo: center=1.0, 3x3 neighborhood=0.5
                # Set center to 1.0 (type-safe assignment)
                if isinstance(cls_targets, np.ndarray):
                    cls_targets[cls_idx, cy_int, cx_int] = np.float32(1.0)
                elif isinstance(cls_targets, torch.Tensor):
                    cls_targets[cls_idx, cy_int, cx_int] = torch.tensor(1.0, device=cls_targets.device, dtype=cls_targets.dtype)
                else:
                    raise TypeError(f"Unsupported cls_targets type: {type(cls_targets)}")
                # Set 3x3 neighborhood (excluding center) to 0.5
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue  # Skip center (already set to 1.0)
                        ny, nx = cy_int + dy, cx_int + dx
                        # Bounds check
                        if 0 <= ny < cfg.bev_h and 0 <= nx < cfg.bev_w:
                            # Use appropriate maximum based on array type (numpy or torch)
                            val = cls_targets[cls_idx, ny, nx]
                            if isinstance(cls_targets, np.ndarray):
                                cls_targets[cls_idx, ny, nx] = np.maximum(val, np.float32(0.5))
                            elif isinstance(cls_targets, torch.Tensor):
                                cls_targets[cls_idx, ny, nx] = torch.maximum(
                                    val, torch.tensor(0.5, device=val.device, dtype=val.dtype)
                                )
                            else:
                                raise TypeError(f"Unsupported cls_targets type: {type(cls_targets)}")
            else:
                raise ValueError(f"Unknown heatmap_mode: {cfg.heatmap_mode}. "
                               f"Must be one of: 'gaussian', 'hard_center', 'hard_center_radius1'")

        # Validate targets at DEBUG level
        u = np.unique(cls_targets)
        logger.debug(f"Target generation complete: heatmap_mode={cfg.heatmap_mode}, unique_values={u.tolist()}")

        # Safety check: hard_center must be strictly binary
        if cfg.heatmap_mode == "hard_center":
            if not set(u.tolist()) <= {0.0, 1.0}:
                raise RuntimeError(
                    f"hard_center target not binary! unique={u.tolist()}. "
                    f"This indicates a bug in target generation."
                )
        if cfg.heatmap_mode == "hard_center_radius1":
            if not set(u.tolist()) <= {0.0, 0.5, 1.0}:
                raise RuntimeError(f"hard_center_radius1 unexpected unique values: {u.tolist()}")

        return {
            'cls_targets': torch.from_numpy(cls_targets),
            'bbox_targets': torch.from_numpy(bbox_targets),
            'reg_mask': torch.from_numpy(reg_mask),
            'indices': torch.from_numpy(indices),
            'num_objs': num_objs
        }

    def _get_class_index(self, label: str) -> Optional[int]:
        """
        Get class index from label string.

        Handles nuScenes class hierarchy (e.g., 'vehicle.car' -> 'car').

        Args:
            label: Class label string

        Returns:
            Class index or None if not found
        """
        # Handle hierarchical labels (e.g., 'vehicle.car')
        label_lower = label.lower()

        # Direct match
        if label_lower in CLASS_TO_IDX:
            return CLASS_TO_IDX[label_lower]

        # Try suffix matching for hierarchical labels
        for cls_name, idx in CLASS_TO_IDX.items():
            if label_lower.endswith(cls_name):
                return idx
            if cls_name in label_lower:
                return idx

        return None

    def _meters_to_pixels(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert world coordinates (meters) to BEV pixel coordinates.

        Uses pixel-center semantics to match the camera encoder's BEV grid:
        - Pixel i covers the range [x_min + i*x_res, x_min + (i+1)*x_res)
        - Pixel i's center is at x_min + (i + 0.5) * x_res
        - An object at x maps to pixel index (x - x_min) / x_res

        This ensures that when an object is at the center of pixel i,
        the sub-pixel offset (dx, dy) will be approximately 0.5, which
        correctly represents the offset from pixel origin to pixel center.

        Args:
            x: X coordinate in meters
            y: Y coordinate in meters

        Returns:
            (px_x, px_y) pixel coordinates (can be fractional)
        """
        px_x = (x - self.config.x_range[0]) / self.x_res
        px_y = (y - self.config.y_range[0]) / self.y_res
        return px_x, px_y

    def _is_in_bounds(self, px_x: float, px_y: float) -> bool:
        """
        Check if pixel coordinates are within BEV bounds.

        Args:
            px_x: X pixel coordinate
            px_y: Y pixel coordinate

        Returns:
            True if within bounds
        """
        return (0 <= px_x < self.config.bev_w and
                0 <= px_y < self.config.bev_h)


def generate_targets_batch(
    annotations_batch: List[List[Box3D]],
    config: Optional[TargetConfig] = None
) -> Dict[str, torch.Tensor]:
    """
    Generate targets for a batch of samples.

    Args:
        annotations_batch: List of annotation lists, one per sample
        config: Target generation configuration

    Returns:
        Batched dictionary of targets
    """
    generator = TargetGenerator(config)

    batch_cls = []
    batch_bbox = []
    batch_mask = []

    for annotations in annotations_batch:
        targets = generator.generate(annotations)
        batch_cls.append(targets['cls_targets'])
        batch_bbox.append(targets['bbox_targets'])
        batch_mask.append(targets['reg_mask'])

    return {
        'cls_targets': torch.stack(batch_cls, dim=0),
        'bbox_targets': torch.stack(batch_bbox, dim=0),
        'reg_mask': torch.stack(batch_mask, dim=0)
    }
