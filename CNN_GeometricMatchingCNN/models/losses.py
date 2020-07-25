# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

from models.geo_transform import PointTranform

class TransformedGridLoss(nn.Module):
    def __init__(self, device = torch.device("cuda"), geometric_model = 'affine', grid_size = 20 ):
        super(TransformedGridLoss, self).__init__()
        self.device = device
        self.geometric_model = geometric_model
        self.point_transform = PointTranform(self.device)
        
        # define virtual grid of points to be transformed
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False).to(self.device)
        return

    def forward(self, theta, image_gt):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)

        # compute transformed grid points using estimated and GT tnfs
        if( self.geometric_model == 'affine' ):
            P_prime = self.point_transform.transform_affine(theta,P)
            P_prime_gt = self.point_transform.transform_affine(image_gt,P)
        elif( self.geometric_model == 'tps' ):
            P_prime = self.point_transform.transform_tps(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_gt = self.point_transform.transform_tps(image_gt,P)
        elif( self.geometric_model == 'hom' ):
            P_prime = self.point_transform.transform_hom(theta,P)
            P_prime_gt = self.point_transform.transform_hom(image_gt,P)
        else:
            NotImplementedError()

        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_gt,2),1)
        loss = torch.mean(loss)
        return loss