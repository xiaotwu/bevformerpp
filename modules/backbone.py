import torch
import torch.nn as nn
import timm

class ResNetBackbone(nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True, out_indices=(1, 2, 3, 4)):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=out_indices)
        self.out_channels = self.backbone.feature_info.channels()

    def forward(self, x):
        # x: (B * Seq * N_cam, 3, H, W)
        features = self.backbone(x)
        return features
