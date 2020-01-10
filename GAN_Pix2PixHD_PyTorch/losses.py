#coding=utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMatchingLoss(nn.Module):
    """
    Pix2Pix-HD の Feature Matching Loss
    """
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        return

    def forward(self, d_input, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            zeros_tsr =  torch.zeros( d_input.shape ).cuda()
            loss_real = - torch.mean( torch.min(d_input - 1, zeros_tsr) )
            loss_fake = - torch.mean( torch.min(-d_input - 1, zeros_tsr) )
            loss = loss_real + loss_fake

        # Generator 用の loss
        else:
            loss = - torch.mean(d_input)

        return loss