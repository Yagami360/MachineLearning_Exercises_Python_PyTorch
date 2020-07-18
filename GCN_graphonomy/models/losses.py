# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParsingCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ParsingCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        return

    def forward(self, input, target):
        # input
        input = input.transpose(0, 1)
        c = input.size()[0]
        n = input.size()[1] * input.size()[2] * input.size()[3]
        input = input.contiguous().view(c, n)
        input = input.transpose(0, 1)

        # target
        [_, argmax] = target.max(dim=1)
        target = argmax.view(n)

        #print( "input.shape={}, target.shape={}".format(input.shape, target.shape) )
        return self.loss_fn(input, target)