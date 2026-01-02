"""
Detection head for 3D bounding box prediction from BEV features.
Implements dense prediction with classification and regression heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .data_structures import Box3D


class DetectionHead(nn.Module):
    """
    3D detection head for BEV features.
    
    Predicts:
    - Classification scores for object categories
    - Regression parameters for 3D bounding boxes (x, y, z, w, l, h, yaw)
    
    Implements Requirements 6.1-6.5 from the design document.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        shared_channels: int = 256,
        num_shared_convs: int = 2
    ):
        """
        Initialize detection head.
        
        Args:
            in_channels: Number of input feature channels
            num_classes: Number of object categories
            shared_channels: Number of channels in shared conv layers
            num_shared_convs: Number of shared convolutional layers
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.shared_channels = shared_channels
        
        # Shared convolutional layers for feature refinement
        shared_layers = []
        for i in range(num_shared_convs):
            in_ch = in_channels if i == 0 else shared_channels
            shared_layers.extend([
                nn.Conv2d(in_ch, shared_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(shared_channels),
                nn.ReLU(inplace=True)
            ])
        self.shared_conv = nn.Sequential(*shared_layers)
        
        # Classification head
        self.cls_head = nn.Conv2d(shared_channels, num_classes, kernel_size=1)
        
        # Regression head (7 parameters: x, y, z, w, l, h, yaw)
        self.reg_head = nn.Conv2d(shared_channels, 7, kernel_size=1)
        
        # Initialize classification head bias for stability
        # Use prior probability of 0.1 (less conservative than 0.01) to allow
        # faster learning of object locations while still suppressing background.
        # Too low (0.01) makes initial outputs near-zero everywhere, slowing convergence.
        prior_prob = 0.1
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)
        
        # Initialize regression head with small weights
        nn.init.normal_(self.reg_head.weight, std=0.001)
        nn.init.constant_(self.reg_head.bias, 0)
    
    def forward(self, bev_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of detection head.
        
        Args:
            bev_features: BEV features, shape (B, C, H, W)
        
        Returns:
            Dictionary containing:
            - 'cls_scores': Classification scores, shape (B, num_classes, H, W)
            - 'bbox_preds': Bounding box parameters, shape (B, 7, H, W)
        """
        # Apply shared convolutions
        shared_features = self.shared_conv(bev_features)
        
        # Classification prediction (return raw logits, not probabilities)
        cls_scores = self.cls_head(shared_features)
        
        # Regression prediction
        bbox_preds = self.reg_head(shared_features)
        
        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds
        }


class BEVHead(nn.Module):
    """Legacy class name for backward compatibility."""
    
    def __init__(self, embed_dim=256, num_classes=10, reg_channels=8):
        super().__init__()
        self.detection_head = DetectionHead(
            in_channels=embed_dim,
            num_classes=num_classes
        )
        
        # Init bias for cls head - use prior_prob=0.1 for faster convergence
        prior_prob = 0.1
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.detection_head.cls_head.bias, bias_value)

    def forward(self, bev_features):
        """
        bev_features: (B, C, H, W)
        """
        outputs = self.detection_head(bev_features)
        
        # Rename for backward compatibility
        return {
            'cls_score': outputs['cls_scores'],
            'bbox_pred': outputs['bbox_preds']
        }

class DetectionLoss(nn.Module):
    """
    Detection loss combining classification (BCE with Logits) and
    regression (Smooth L1 / L1 Loss) for CenterNet-style 3D object detection.

    Classification loss operates in logits space using BCEWithLogitsLoss for
    numerical stability and better convergence. Uses pos_weight to handle
    extreme class imbalance (sparse object centers).

    This loss function is designed to work with the target generator which produces:
    - cls_targets: Binary or Gaussian heatmaps for object centers
    - bbox_targets: 7-channel regression targets (dx, dy, z, log_w, log_l, log_h, yaw)
    - reg_mask: Binary mask indicating valid regression locations

    Reference:
        - CenterNet: Objects as Points (Zhou et al., 2019)
        - CenterPoint: Center-based 3D Object Detection (Yin et al., 2021)
    """

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0,
                 use_smooth_l1: bool = True, smooth_l1_beta: float = 1.0,
                 pos_weight: float = 50.0, neg_weight: float = 2.0):
        """
        Initialize detection loss.

        Args:
            cls_weight: Weight for classification loss (default: 1.0)
            reg_weight: Weight for regression loss (default: 1.0)
            use_smooth_l1: Use Smooth L1 instead of L1 loss (default: True)
            smooth_l1_beta: Beta parameter for Smooth L1 (default: 1.0)
            pos_weight: Weight for positive samples in BCE loss (default: 50.0)
            neg_weight: Multiplier for negative sample loss to increase background suppression (default: 2.0)

        Note: reg_weight was increased from 0.1 to 1.0 to provide stronger
        regression supervision at object centers, which helps the classification
        head learn better object/background separation.
        """
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.use_smooth_l1 = use_smooth_l1
        self.smooth_l1_beta = smooth_l1_beta
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

        if use_smooth_l1:
            self.reg_loss_fn = nn.SmoothL1Loss(reduction='none', beta=smooth_l1_beta)
        else:
            self.reg_loss_fn = nn.L1Loss(reduction='none')

    def forward(self, preds: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.

        Args:
            preds: Model predictions containing:
                - 'cls_score' or 'cls_scores': (B, K, H, W) classification heatmaps
                - 'bbox_pred' or 'bbox_preds': (B, 7, H, W) regression outputs

            targets: Ground truth targets containing:
                - 'cls_targets' or 'gt_cls': (B, K, H, W) Gaussian heatmaps
                - 'bbox_targets' or 'gt_bbox': (B, 7, H, W) regression targets
                - 'reg_mask' or 'mask': (B, 1, H, W) valid regression mask

        Returns:
            Dictionary containing:
                - 'loss_total': Combined weighted loss
                - 'loss_cls': Classification loss
                - 'loss_bbox': Regression loss
                - 'num_pos': Number of positive samples (for logging)
        """
        # Handle different key naming conventions (backward compatibility)
        cls_pred = preds.get('cls_score', preds.get('cls_scores'))
        bbox_pred = preds.get('bbox_pred', preds.get('bbox_preds'))

        gt_cls = targets.get('cls_targets', targets.get('gt_cls'))
        gt_bbox = targets.get('bbox_targets', targets.get('gt_bbox'))
        mask = targets.get('reg_mask', targets.get('mask'))

        # Validate shapes
        assert cls_pred is not None, "Missing classification predictions"
        assert bbox_pred is not None, "Missing bbox predictions"
        assert gt_cls is not None, "Missing classification targets"
        assert gt_bbox is not None, "Missing bbox targets"
        assert mask is not None, "Missing regression mask"

        # Ensure mask is broadcastable to bbox shape
        if mask.shape[1] == 1 and bbox_pred.shape[1] > 1:
            mask = mask.expand_as(bbox_pred)

        # 1. Classification Loss (BCE with Logits in logits space)
        loss_cls = self.bce_with_logits_loss(cls_pred, gt_cls)

        # 2. Regression Loss (Masked L1/SmoothL1)
        loss_bbox_raw = self.reg_loss_fn(bbox_pred, gt_bbox)
        num_pos = mask.sum().clamp(min=1.0)  # Avoid division by zero
        loss_bbox = (loss_bbox_raw * mask).sum() / num_pos

        # Combined loss
        loss_total = self.cls_weight * loss_cls + self.reg_weight * loss_bbox

        return {
            'loss_total': loss_total,
            'loss_cls': loss_cls,
            'loss_bbox': loss_bbox,
            'num_pos': num_pos
        }

    def bce_with_logits_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Binary Cross-Entropy with Logits Loss for heatmap-based object detection.

        Operates in logits space for numerical stability and better convergence.
        Uses pos_weight to handle extreme class imbalance (sparse object centers).

        Args:
            logits: Predicted logits, shape (B, K, H, W)
            target: Target heatmaps, shape (B, K, H, W), values in [0, 1]

        Returns:
            Scalar BCE loss value (normalized by number of positive samples)
        """
        # Compute BCE with logits (numerically stable, operates in logits space)
        # pos_weight scales positive examples to handle class imbalance
        # Convert pos_weight to tensor with shape (1, K, 1, 1) for proper broadcasting
        B, K, H, W = logits.shape
        # Create tensor with shape (1, K, 1, 1) to match spatial dimensions
        pos_weight_tensor = torch.full(
            (1, K, 1, 1), self.pos_weight, dtype=logits.dtype, device=logits.device
        )
        
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, target, reduction='none', pos_weight=pos_weight_tensor
        )
        
        # Identify positive and negative samples
        pos_inds = target.eq(1.0).float()
        neg_inds = target.eq(0.0).float()
        
        # Separate positive and negative losses
        pos_loss = (bce_loss * pos_inds).sum()
        neg_loss = (bce_loss * neg_inds).sum()
        
        # Apply negative weight multiplier to increase background suppression
        neg_loss = neg_loss * self.neg_weight

        # Normalize by number of positive samples
        num_pos = pos_inds.sum()
        if num_pos == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / num_pos

        return loss



def compute_iou_bev_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute BEV IoU between two sets of bounding boxes (vectorized).

    Args:
        boxes1: Tensor of shape (N, 4) with [x, y, w, l] for each box
        boxes2: Tensor of shape (M, 4) with [x, y, w, l] for each box

    Returns:
        IoU matrix of shape (N, M)
    """
    # Extract box parameters
    x1, y1, w1, l1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x2, y2, w2, l2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # Compute half sizes
    half_w1, half_l1 = w1 / 2, l1 / 2
    half_w2, half_l2 = w2 / 2, l2 / 2

    # Box bounds: (N,) and (M,)
    x1_min, x1_max = x1 - half_w1, x1 + half_w1
    y1_min, y1_max = y1 - half_l1, y1 + half_l1
    x2_min, x2_max = x2 - half_w2, x2 + half_w2
    y2_min, y2_max = y2 - half_l2, y2 + half_l2

    # Compute pairwise intersection: (N, M)
    inter_x_min = torch.maximum(x1_min.unsqueeze(1), x2_min.unsqueeze(0))
    inter_x_max = torch.minimum(x1_max.unsqueeze(1), x2_max.unsqueeze(0))
    inter_y_min = torch.maximum(y1_min.unsqueeze(1), y2_min.unsqueeze(0))
    inter_y_max = torch.minimum(y1_max.unsqueeze(1), y2_max.unsqueeze(0))

    inter_w = (inter_x_max - inter_x_min).clamp(min=0)
    inter_h = (inter_y_max - inter_y_min).clamp(min=0)
    inter_area = inter_w * inter_h

    # Compute areas
    area1 = w1 * l1  # (N,)
    area2 = w2 * l2  # (M,)

    # Union: (N, M)
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def compute_iou_3d(box1: Box3D, box2: Box3D) -> float:
    """
    Compute 3D IoU between two bounding boxes.
    Simplified version using BEV IoU (2D intersection over union).

    Args:
        box1: First bounding box
        box2: Second bounding box

    Returns:
        IoU value in [0, 1]
    """
    # Extract 2D box parameters (x, y, w, l, yaw)
    x1, y1 = box1.center[0], box1.center[1]
    w1, l1 = box1.size[0], box1.size[1]

    x2, y2 = box2.center[0], box2.center[1]
    w2, l2 = box2.size[0], box2.size[1]

    # Simplified BEV IoU using axis-aligned approximation
    # Compute axis-aligned bounding boxes
    half_w1, half_l1 = w1 / 2, l1 / 2
    half_w2, half_l2 = w2 / 2, l2 / 2

    # Box 1 bounds
    x1_min, x1_max = x1 - half_w1, x1 + half_w1
    y1_min, y1_max = y1 - half_l1, y1 + half_l1

    # Box 2 bounds
    x2_min, x2_max = x2 - half_w2, x2 + half_w2
    y2_min, y2_max = y2 - half_l2, y2 + half_l2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_min = max(y1_min, y2_min)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Union
    area1 = w1 * l1
    area2 = w2 * l2
    union_area = area1 + area2 - inter_area

    iou = inter_area / (union_area + 1e-6)
    return float(iou)


def nms_3d(boxes: List[Box3D], iou_threshold: float = 0.5) -> List[Box3D]:
    """
    Apply Non-Maximum Suppression to 3D bounding boxes (optimized).

    Uses vectorized IoU computation for better performance.

    Args:
        boxes: List of Box3D objects with confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered list of Box3D objects after NMS
    """
    if len(boxes) == 0:
        return []

    # Group boxes by class for per-class NMS
    class_to_boxes = {}
    for i, box in enumerate(boxes):
        if box.label not in class_to_boxes:
            class_to_boxes[box.label] = []
        class_to_boxes[box.label].append((i, box))

    keep_indices = []

    for label, class_boxes in class_to_boxes.items():
        if len(class_boxes) == 0:
            continue

        # Sort by score (descending)
        class_boxes.sort(key=lambda x: x[1].score, reverse=True)
        indices = [x[0] for x in class_boxes]
        class_box_objs = [x[1] for x in class_boxes]

        n = len(class_box_objs)

        # Build tensor for vectorized IoU: [x, y, w, l]
        box_params = torch.tensor([
            [b.center[0], b.center[1], b.size[0], b.size[1]]
            for b in class_box_objs
        ], dtype=torch.float32)

        # Compute pairwise IoU matrix once
        iou_matrix = compute_iou_bev_batch(box_params, box_params)

        # Greedy NMS using precomputed IoU
        suppressed = torch.zeros(n, dtype=torch.bool)
        for i in range(n):
            if suppressed[i]:
                continue

            keep_indices.append(indices[i])

            # Suppress all boxes with high IoU
            suppressed = suppressed | (iou_matrix[i] > iou_threshold)
            suppressed[i] = False  # Don't suppress the kept box for next iteration check

    return [boxes[i] for i in keep_indices]


def nms_3d_vectorized(
    boxes_tensor: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    max_detections: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fully vectorized NMS operating on tensors (GPU-accelerated).

    Args:
        boxes_tensor: Box parameters (N, 7) with [x, y, z, w, l, h, yaw]
        scores: Confidence scores (N,)
        labels: Class labels (N,)
        iou_threshold: IoU threshold for suppression
        max_detections: Maximum detections to keep

    Returns:
        Tuple of (boxes, scores, labels) for kept detections
    """
    if boxes_tensor.shape[0] == 0:
        return boxes_tensor, scores, labels

    device = boxes_tensor.device

    # Get unique classes
    unique_labels = labels.unique()
    keep_mask = torch.zeros(boxes_tensor.shape[0], dtype=torch.bool, device=device)

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_indices = torch.where(cls_mask)[0]
        cls_scores = scores[cls_mask]
        cls_boxes = boxes_tensor[cls_mask]

        if len(cls_indices) == 0:
            continue

        # Sort by score
        sorted_indices = torch.argsort(cls_scores, descending=True)
        cls_boxes = cls_boxes[sorted_indices]
        cls_indices = cls_indices[sorted_indices]

        # Build IoU tensor: [x, y, w, l] from [x, y, z, w, l, h, yaw]
        box_params = cls_boxes[:, [0, 1, 3, 4]]  # x, y, w, l
        iou_matrix = compute_iou_bev_batch(box_params, box_params)

        # Greedy NMS
        n = cls_boxes.shape[0]
        suppressed = torch.zeros(n, dtype=torch.bool, device=device)

        for i in range(n):
            if suppressed[i]:
                continue

            keep_mask[cls_indices[i]] = True

            # Suppress overlapping boxes
            suppressed = suppressed | (iou_matrix[i] > iou_threshold)
            suppressed[i] = False

    # Get kept boxes
    kept_boxes = boxes_tensor[keep_mask]
    kept_scores = scores[keep_mask]
    kept_labels = labels[keep_mask]

    # Sort by score and limit
    if kept_boxes.shape[0] > max_detections:
        top_indices = torch.argsort(kept_scores, descending=True)[:max_detections]
        kept_boxes = kept_boxes[top_indices]
        kept_scores = kept_scores[top_indices]
        kept_labels = kept_labels[top_indices]

    return kept_boxes, kept_scores, kept_labels


def _extract_peaks(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Extract local maxima (peaks) from heatmap using max pooling.

    This is much faster than iterating over all pixels above threshold.

    Args:
        heatmap: Heatmap tensor of shape (B, C, H, W)
        kernel_size: Size of max pooling kernel

    Returns:
        Peak mask of same shape as input
    """
    padding = (kernel_size - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=padding)
    peaks = (hmax == heatmap).float()
    return peaks * heatmap


def decode_detections_fast(
    cls_scores: torch.Tensor,
    bbox_preds: torch.Tensor,
    score_threshold: float = 0.3,
    max_per_class: int = 50,
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
    class_names: Optional[List[str]] = None,
    use_peak_detection: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Decode detection predictions to tensors (optimized, GPU-accelerated).

    Uses local maxima detection instead of thresholding every pixel.

    Args:
        cls_scores: Classification scores, shape (B, num_classes, H, W)
        bbox_preds: Bounding box predictions, shape (B, 7, H, W)
        score_threshold: Confidence threshold for filtering
        max_per_class: Maximum detections per class per batch
        bev_range: BEV range (x_min, x_max, y_min, y_max)
        class_names: List of class names (unused, for compatibility)
        use_peak_detection: Use local maxima detection

    Returns:
        Tuple of (boxes, scores, labels, batch_indices) tensors
        - boxes: (N, 7) box parameters
        - scores: (N,) confidence scores
        - labels: (N,) class labels
        - batch_indices: (N,) batch indices (all 0 for single batch)
    """
    B, num_classes, H, W = cls_scores.shape
    device = cls_scores.device

    x_min, x_max, y_min, y_max = bev_range
    resolution_x = (x_max - x_min) / W
    resolution_y = (y_max - y_min) / H

    # Apply peak detection (local maxima)
    if use_peak_detection:
        peaks = _extract_peaks(cls_scores, kernel_size=3)
    else:
        peaks = cls_scores

    # Apply threshold
    peaks = peaks * (peaks > score_threshold).float()

    all_boxes = []
    all_scores = []
    all_labels = []
    all_batch_indices = []  # Track which batch each detection belongs to

    for b in range(B):
        for cls_idx in range(num_classes):
            cls_map = peaks[b, cls_idx]  # (H, W)

            # Get top-k scores for this class
            flat_scores = cls_map.view(-1)
            k = min(max_per_class, (flat_scores > 0).sum().item())

            if k == 0:
                continue

            topk_scores, topk_indices = torch.topk(flat_scores, k)

            # Filter by threshold
            valid_mask = topk_scores > score_threshold
            if not valid_mask.any():
                continue

            topk_scores = topk_scores[valid_mask]
            topk_indices = topk_indices[valid_mask]

            # Convert flat indices to 2D
            h_indices = topk_indices // W
            w_indices = topk_indices % W

            # Get bbox parameters at valid locations
            n_valid = len(topk_indices)
            bbox_params = bbox_preds[b, :, h_indices, w_indices]  # (7, n_valid)

            # Decode box parameters (vectorized)
            x = x_min + (w_indices.float() + 0.5) * resolution_x + bbox_params[0]
            y = y_min + (h_indices.float() + 0.5) * resolution_y + bbox_params[1]
            z = bbox_params[2]
            w = torch.exp(bbox_params[3].clamp(max=10))  # Clamp to avoid explosion
            l = torch.exp(bbox_params[4].clamp(max=10))
            h = torch.exp(bbox_params[5].clamp(max=10))
            yaw = bbox_params[6]

            # Stack boxes: (n_valid, 7)
            boxes = torch.stack([x, y, z, w, l, h, yaw], dim=1)
            labels = torch.full((n_valid,), cls_idx, dtype=torch.long, device=device)
            batch_indices = torch.full((n_valid,), b, dtype=torch.long, device=device)

            all_boxes.append(boxes)
            all_scores.append(topk_scores)
            all_labels.append(labels)
            all_batch_indices.append(batch_indices)

    if len(all_boxes) == 0:
        return (
            torch.zeros((0, 7), device=device),
            torch.zeros((0,), device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device)  # batch_indices (all 0 for single batch)
        )

    return (
        torch.cat(all_boxes, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_batch_indices, dim=0)  # Return batch indices for proper evaluation
    )


def decode_detections(
    cls_scores: torch.Tensor,
    bbox_preds: torch.Tensor,
    score_threshold: float = 0.3,
    bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
    class_names: Optional[List[str]] = None
) -> List[Box3D]:
    """
    Decode detection predictions to Box3D objects.
    
    Args:
        cls_scores: Classification scores, shape (B, num_classes, H, W)
        bbox_preds: Bounding box predictions, shape (B, 7, H, W)
        score_threshold: Confidence threshold for filtering
        bev_range: BEV range (x_min, x_max, y_min, y_max)
        class_names: List of class names
    
    Returns:
        List of Box3D objects
    """
    B, num_classes, H, W = cls_scores.shape
    device = cls_scores.device
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    x_min, x_max, y_min, y_max = bev_range
    resolution_x = (x_max - x_min) / W
    resolution_y = (y_max - y_min) / H
    
    all_boxes = []
    
    # Process each batch
    for b in range(B):
        batch_boxes = []
        
        # Find peaks in classification heatmap
        for cls_idx in range(num_classes):
            cls_map = cls_scores[b, cls_idx]  # (H, W)
            
            # Find locations above threshold
            valid_mask = cls_map > score_threshold
            valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # (N, 2)
            
            if valid_indices.shape[0] == 0:
                continue
            
            # Extract predictions at valid locations
            for idx in valid_indices:
                h_idx, w_idx = idx[0].item(), idx[1].item()
                
                # Get confidence score
                score = cls_map[h_idx, w_idx].item()
                
                # Get bounding box parameters
                bbox_params = bbox_preds[b, :, h_idx, w_idx].cpu().numpy()
                
                # Decode box parameters
                # Center position in BEV coordinates
                x = x_min + (w_idx + 0.5) * resolution_x + bbox_params[0]
                y = y_min + (h_idx + 0.5) * resolution_y + bbox_params[1]
                z = bbox_params[2]
                
                # Size
                w = np.exp(bbox_params[3])  # Predict log(w)
                l = np.exp(bbox_params[4])  # Predict log(l)
                h = np.exp(bbox_params[5])  # Predict log(h)
                
                # Yaw angle
                yaw = bbox_params[6]
                
                # Create Box3D
                box = Box3D(
                    center=np.array([x, y, z], dtype=np.float32),
                    size=np.array([w, l, h], dtype=np.float32),
                    yaw=float(yaw),
                    label=class_names[cls_idx],
                    score=float(score)
                )
                
                batch_boxes.append(box)
        
        all_boxes.extend(batch_boxes)
    
    return all_boxes


class DetectionPostProcessor(nn.Module):
    """
    Post-processing module for detection outputs.
    Applies NMS and decoding to produce final detections.

    Optimized version using vectorized operations for GPU acceleration.
    """

    def __init__(
        self,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        max_per_class: int = 50,
        bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
        class_names: Optional[List[str]] = None,
        use_fast_decode: bool = True
    ):
        """
        Initialize post-processor.

        Args:
            score_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep
            max_per_class: Maximum detections per class before NMS
            bev_range: BEV range (x_min, x_max, y_min, y_max)
            class_names: List of class names
            use_fast_decode: Use optimized decoding with peak detection
        """
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.max_per_class = max_per_class
        self.bev_range = bev_range
        self.class_names = class_names
        self.use_fast_decode = use_fast_decode

    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor
    ) -> List[List[Box3D]]:
        """
        Apply post-processing to detection outputs.

        Args:
            cls_scores: Classification scores, shape (B, num_classes, H, W)
            bbox_preds: Bounding box predictions, shape (B, 7, H, W)

        Returns:
            List of Box3D lists, one per sample in batch: List[List[Box3D]]
        """
        B = cls_scores.shape[0]
        batch_results = []
        
        if self.use_fast_decode:
            # Process each sample in batch separately to track which predictions belong to which sample
            for b in range(B):
                # Extract single sample
                sample_cls = cls_scores[b:b+1]  # (1, num_classes, H, W)
                sample_bbox = bbox_preds[b:b+1]  # (1, 7, H, W)
                
                # Decode detections for this sample
                boxes_tensor, scores, labels, batch_indices = decode_detections_fast(
                    sample_cls,
                    sample_bbox,
                    score_threshold=self.score_threshold,
                    max_per_class=self.max_per_class,
                    bev_range=self.bev_range,
                    use_peak_detection=True
                )

                # Apply vectorized NMS
                boxes_tensor, scores, labels = nms_3d_vectorized(
                    boxes_tensor,
                    scores,
                    labels,
                    iou_threshold=self.nms_threshold,
                    max_detections=self.max_detections
                )

                # Convert tensors to Box3D list for this sample
                sample_boxes = self._tensor_to_box3d(boxes_tensor, scores, labels)
                batch_results.append(sample_boxes)
        else:
            # Legacy path (slower) - process per sample
            for b in range(B):
                sample_cls = cls_scores[b:b+1]
                sample_bbox = bbox_preds[b:b+1]
                
                boxes = decode_detections(
                    sample_cls,
                    sample_bbox,
                    score_threshold=self.score_threshold,
                    bev_range=self.bev_range,
                    class_names=self.class_names
                )
                boxes = nms_3d(boxes, iou_threshold=self.nms_threshold)

                if len(boxes) > self.max_detections:
                    boxes = sorted(boxes, key=lambda x: x.score, reverse=True)
                    boxes = boxes[:self.max_detections]

                batch_results.append(boxes)

        return batch_results

    def _tensor_to_box3d(
        self,
        boxes_tensor: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor
    ) -> List[Box3D]:
        """Convert tensor outputs to Box3D objects."""
        if boxes_tensor.shape[0] == 0:
            return []

        boxes_np = boxes_tensor.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        class_names = self.class_names or [f"class_{i}" for i in range(10)]

        result = []
        for i in range(len(boxes_np)):
            box = Box3D(
                center=np.array([boxes_np[i, 0], boxes_np[i, 1], boxes_np[i, 2]], dtype=np.float32),
                size=np.array([boxes_np[i, 3], boxes_np[i, 4], boxes_np[i, 5]], dtype=np.float32),
                yaw=float(boxes_np[i, 6]),
                label=class_names[labels_np[i]] if labels_np[i] < len(class_names) else f"class_{labels_np[i]}",
                score=float(scores_np[i])
            )
            result.append(box)

        return result
