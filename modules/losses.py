"""
Loss functions for BEV Fusion System.
Implements classification loss (focal loss) and regression loss (smooth L1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in dense object detection.
    Based on "Focal Loss for Dense Object Detection" (Lin et al., 2017).
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        """
        Initialize focal loss.
        
        Args:
            alpha: Focusing parameter (default: 2.0)
            beta: Modulating parameter for negative samples (default: 4.0)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            pred: Predicted scores, shape (B, num_classes, H, W), values in [0, 1]
            target: Target heatmap, shape (B, num_classes, H, W), values in [0, 1]
        
        Returns:
            Scalar loss value
        """
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-4, max=1 - 1e-4)
        
        # Positive and negative sample masks
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        # Negative sample weights (Gaussian weighting)
        neg_weights = torch.pow(1 - target, self.beta)
        
        # Positive loss
        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        
        # Negative loss
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds
        
        # Normalize by number of positive samples
        num_pos = pos_inds.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        
        return loss


class SmoothL1Loss(nn.Module):
    """
    Smooth L1 loss for bounding box regression.
    Less sensitive to outliers than L2 loss.
    """
    
    def __init__(self, beta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize smooth L1 loss.
        
        Args:
            beta: Threshold for switching between L1 and L2 (default: 1.0)
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute smooth L1 loss.
        
        Args:
            pred: Predicted values, shape (B, C, H, W)
            target: Target values, shape (B, C, H, W)
            mask: Optional mask for valid samples, shape (B, 1, H, W)
        
        Returns:
            Loss value
        """
        diff = torch.abs(pred - target)
        
        # Smooth L1: L2 for small errors, L1 for large errors
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta
        )
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
            if self.reduction == 'mean':
                return loss.sum() / (mask.sum() + 1e-6)
            elif self.reduction == 'sum':
                return loss.sum()
        
        # Standard reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DetectionLoss(nn.Module):
    """
    Combined loss for 3D object detection.
    Includes classification loss and regression loss with configurable weights.
    
    Implements Requirements 8.2 from the design document.
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
        focal_alpha: float = 2.0,
        focal_beta: float = 4.0,
        smooth_l1_beta: float = 1.0
    ):
        """
        Initialize detection loss.
        
        Args:
            cls_weight: Weight for classification loss
            reg_weight: Weight for regression loss
            focal_alpha: Focal loss alpha parameter
            focal_beta: Focal loss beta parameter
            smooth_l1_beta: Smooth L1 loss beta parameter
        """
        super().__init__()
        
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        
        # Classification loss (focal loss)
        self.focal_loss = FocalLoss(alpha=focal_alpha, beta=focal_beta)
        
        # Regression loss (smooth L1)
        self.smooth_l1_loss = SmoothL1Loss(beta=smooth_l1_beta, reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute detection loss.
        
        Args:
            predictions: Dictionary containing:
                - 'cls_scores': Classification scores, shape (B, num_classes, H, W)
                - 'bbox_preds': Bounding box predictions, shape (B, 7, H, W)
            targets: Dictionary containing:
                - 'cls_targets': Target heatmap, shape (B, num_classes, H, W)
                - 'bbox_targets': Target boxes, shape (B, 7, H, W)
                - 'reg_mask': Mask for valid regression targets, shape (B, 1, H, W)
        
        Returns:
            Dictionary containing:
                - 'loss_total': Total loss
                - 'loss_cls': Classification loss
                - 'loss_reg': Regression loss
        """
        # Extract predictions
        cls_scores = predictions['cls_scores']
        bbox_preds = predictions['bbox_preds']
        
        # Extract targets
        cls_targets = targets['cls_targets']
        bbox_targets = targets['bbox_targets']
        reg_mask = targets.get('reg_mask', None)
        
        # Classification loss
        loss_cls = self.focal_loss(cls_scores, cls_targets)
        
        # Regression loss (only for positive samples)
        loss_reg = self.smooth_l1_loss(bbox_preds, bbox_targets, mask=reg_mask)
        
        # Separate losses for different box parameters
        # This can help with debugging and tuning
        if reg_mask is not None:
            # Center loss (x, y, z)
            loss_center = self.smooth_l1_loss(
                bbox_preds[:, :3],
                bbox_targets[:, :3],
                mask=reg_mask
            )
            
            # Size loss (w, l, h)
            loss_size = self.smooth_l1_loss(
                bbox_preds[:, 3:6],
                bbox_targets[:, 3:6],
                mask=reg_mask
            )
            
            # Rotation loss (yaw)
            loss_rot = self.smooth_l1_loss(
                bbox_preds[:, 6:7],
                bbox_targets[:, 6:7],
                mask=reg_mask
            )
        else:
            loss_center = loss_reg
            loss_size = loss_reg
            loss_rot = loss_reg
        
        # Total loss
        loss_total = self.cls_weight * loss_cls + self.reg_weight * loss_reg
        
        return {
            'loss_total': loss_total,
            'loss_cls': loss_cls,
            'loss_reg': loss_reg,
            'loss_center': loss_center,
            'loss_size': loss_size,
            'loss_rot': loss_rot
        }


class GaussianFocalLoss(nn.Module):
    """
    Gaussian Focal Loss for heatmap-based detection (CenterNet style).
    Legacy class for backward compatibility.
    """
    
    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, beta=beta)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(pred, target)
