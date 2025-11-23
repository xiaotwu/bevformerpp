import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channel in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_channel, out_channels, 1))
            self.fpn_convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))

    def forward(self, inputs):
        # inputs: list of features from backbone
        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # Top-down path
        for i in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i], size=prev_shape, mode='bilinear', align_corners=False)

        # Output convs
        outs = [conv(x) for conv, x in zip(self.fpn_convs, laterals)]
        return outs
