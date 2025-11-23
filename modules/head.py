import torch
import torch.nn as nn

class BEVHead(nn.Module):
    def __init__(self, embed_dim=256, num_classes=10, reg_channels=8):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, num_classes, 1)
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, reg_channels, 1)
        )
        
        # Init bias for cls head to prevent instability at start
        self.cls_head[-1].bias.data.fill_(-2.19) # -log((1-pi)/pi) for pi=0.1

    def forward(self, bev_features):
        """
        bev_features: (B, C, H, W)
        """
        cls_score = self.cls_head(bev_features)
        cls_score = torch.sigmoid(cls_score) # Heatmap 0-1
        
        bbox_pred = self.reg_head(bev_features)
        
        return {'cls_score': cls_score, 'bbox_pred': bbox_pred}

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
