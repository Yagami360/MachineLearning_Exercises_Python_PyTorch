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


class CrossEntropy2DLoss(nn.Module):
    def __init__(self, device, ignore_index = 255, weight = None, size_average = True, batch_average = True):
        super(CrossEntropy2DLoss, self).__init__()
        if weight is None:
            self.loss_fn = nn.CrossEntropyLoss( weight = weight, ignore_index = ignore_index, size_average = size_average )
        else:
            self.loss_fn = nn.CrossEntropyLoss( weight = torch.from_numpy(np.array(weight)).float().to(device), ignore_index = ignore_index, size_average = size_average )
        return

    def forward(self, logit, target):
        n, c, h, w = logit.size()
        # logit = logit.permute(0, 2, 3, 1)
        target = target.squeeze(1)

        print( "logit.shape : ", logit.shape )
        print( "target.shape : ", target.shape )
        loss = self.loss_fn(logit, target.long())
        return loss
