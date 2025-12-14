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
        # Use prior probability of 0.01 for rare objects
        prior_prob = 0.01
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
        
        # Classification prediction
        cls_scores = self.cls_head(shared_features)
        cls_scores = torch.sigmoid(cls_scores)  # Apply sigmoid for heatmap
        
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
        
        # Init bias for cls head to prevent instability at start
        prior_prob = 0.01
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
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction='none')
        
    def forward(self, preds, targets):
        """
        preds: {'cls_score': (B, K, H, W), 'bbox_pred': (B, 8, H, W)}
        targets: {'gt_cls': (B, K, H, W), 'gt_bbox': (B, 8, H, W), 'mask': (B, 1, H, W)}
        """
        cls_pred = preds['cls_score']
        bbox_pred = preds['bbox_pred']
        
        gt_cls = targets['gt_cls']
        gt_bbox = targets['gt_bbox']
        mask = targets['mask']
        
        # 1. Classification Loss (Gaussian Focal Loss)
        loss_cls = self.gaussian_focal_loss(cls_pred, gt_cls)
        
        # 2. Regression Loss (L1 Loss, masked)
        loss_bbox = self.l1_loss(bbox_pred, gt_bbox)
        loss_bbox = (loss_bbox * mask).sum() / (mask.sum() + 1e-4)
        
        return {'loss_total': loss_cls + loss_bbox, 'loss_cls': loss_cls, 'loss_bbox': loss_bbox}

    def gaussian_focal_loss(self, pred, target, alpha=2, beta=4):
        """
        Focal Loss for Dense Object Detection (CornerNet/CenterNet style).
        """
        # Clamp pred to avoid log(0)
        pred = torch.clamp(pred, min=1e-4, max=1-1e-4)
        
        pos_inds = target.eq(1)
        neg_inds = target.lt(1)

        neg_weights = torch.pow(1 - target, beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss



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
    yaw1 = box1.yaw
    
    x2, y2 = box2.center[0], box2.center[1]
    w2, l2 = box2.size[0], box2.size[1]
    yaw2 = box2.yaw
    
    # Simplified BEV IoU using axis-aligned approximation
    # For production, use rotated IoU
    
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
    Apply Non-Maximum Suppression to 3D bounding boxes.
    
    Args:
        boxes: List of Box3D objects with confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        Filtered list of Box3D objects after NMS
    """
    if len(boxes) == 0:
        return []
    
    # Sort boxes by confidence score (descending)
    boxes_sorted = sorted(boxes, key=lambda x: x.score, reverse=True)
    
    keep = []
    while len(boxes_sorted) > 0:
        # Keep the box with highest score
        current_box = boxes_sorted[0]
        keep.append(current_box)
        boxes_sorted = boxes_sorted[1:]
        
        # Remove boxes with high IoU
        filtered_boxes = []
        for box in boxes_sorted:
            # Only compare boxes of the same class
            if box.label != current_box.label:
                filtered_boxes.append(box)
                continue
            
            iou = compute_iou_3d(current_box, box)
            if iou < iou_threshold:
                filtered_boxes.append(box)
        
        boxes_sorted = filtered_boxes
    
    return keep


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
    """
    
    def __init__(
        self,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 100,
        bev_range: Tuple[float, float, float, float] = (-51.2, 51.2, -51.2, 51.2),
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize post-processor.
        
        Args:
            score_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections to keep
            bev_range: BEV range (x_min, x_max, y_min, y_max)
            class_names: List of class names
        """
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.bev_range = bev_range
        self.class_names = class_names
    
    def forward(
        self,
        cls_scores: torch.Tensor,
        bbox_preds: torch.Tensor
    ) -> List[Box3D]:
        """
        Apply post-processing to detection outputs.
        
        Args:
            cls_scores: Classification scores, shape (B, num_classes, H, W)
            bbox_preds: Bounding box predictions, shape (B, 7, H, W)
        
        Returns:
            List of Box3D objects after NMS
        """
        # Decode detections
        boxes = decode_detections(
            cls_scores,
            bbox_preds,
            score_threshold=self.score_threshold,
            bev_range=self.bev_range,
            class_names=self.class_names
        )
        
        # Apply NMS
        boxes_nms = nms_3d(boxes, iou_threshold=self.nms_threshold)
        
        # Limit to max detections
        if len(boxes_nms) > self.max_detections:
            boxes_nms = sorted(boxes_nms, key=lambda x: x.score, reverse=True)
            boxes_nms = boxes_nms[:self.max_detections]
        
        return boxes_nms
