import torch
import torch.nn as nn
import torch.nn.functional as F
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', n_in_channels = 1, output_stride=16, num_classes = 1, pretrained_backbone=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = nn.BatchNorm2d
        self.backbone = build_backbone(backbone, n_in_channels, output_stride, BatchNorm, pretrained_backbone)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.activate_tanh = nn.Tanh()
        self.activate_sigmoid = nn.Sigmoid()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        output = self.activate_tanh( x )
        output_mask = self.activate_sigmoid( x )
        return output, output_mask, x
