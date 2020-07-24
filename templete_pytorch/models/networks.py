# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class TempleteNetworks( nn.Module ):
    """
    ダミー用の何もしないネットワーク
    """
    def __init__( self, n_in_channles = 3, n_out_channles = 1 ):
        super( TempleteNetworks, self ).__init__()
        self.dummmy_layer = nn.Sequential(
            nn.Conv2d( n_in_channles, n_out_channles, kernel_size=1, stride=1, padding=0 ),
        )
        return

    def forward( self, input ):
        output = self.dummmy_layer(input)
        return output
