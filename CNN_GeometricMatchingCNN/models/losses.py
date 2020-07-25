# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable

from models.geo_transform import PointTranform

#=============================================
# Transformed Grid Loss
#=============================================
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

    def forward(self, theta, theta_gt):
        # expand grid according to batch size
        batch_size = theta.size()[0]
        P = self.P.expand(batch_size,2,self.N)

        # compute transformed grid points using estimated and GT tnfs
        if( self.geometric_model == 'affine' ):
            P_prime = self.point_transform.transform_affine(theta,P)
            P_prime_gt = self.point_transform.transform_affine(theta_gt,P)
        elif( self.geometric_model == 'tps' ):
            P_prime = self.point_transform.transform_tps(theta.unsqueeze(2).unsqueeze(3),P)
            P_prime_gt = self.point_transform.transform_tps(theta_gt,P)
        elif( self.geometric_model == 'hom' ):
            P_prime = self.point_transform.transform_hom(theta,P)
            P_prime_gt = self.point_transform.transform_hom(theta_gt,P)
        else:
            NotImplementedError()

        # compute MSE loss on transformed grid points
        loss = torch.sum(torch.pow(P_prime - P_prime_gt,2),1)
        loss = torch.mean(loss)
        return loss


#=============================================
# VGG loss
#=============================================
class Vgg19(nn.Module):
    def __init__(self, n_channels=3, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        vgg_pretrained_features[0] = nn.Conv2d( n_channels, 64, kernel_size=3, stride=1, padding=0 )
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, device, n_channels = 3, layids = None ):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19(n_channels=n_channels).to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


#============================================
# GAN Adv loss
#============================================
class VanillaGANLoss(nn.Module):
    def __init__(self, device, w_sigmoid_D = True ):
        super(VanillaGANLoss, self).__init__()
        self.device = device
        # when use sigmoid in Discriminator
        if( w_sigmoid_D ):
            self.loss_fn = nn.BCELoss()            
        # when not use sigmoid in Discriminator
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        return

    def forward_D(self, d_real, d_fake):
        real_ones_tsr = torch.ones( d_real.shape ).to(self.device)
        fake_zeros_tsr = torch.zeros( d_fake.shape ).to(self.device)
        loss_D_real = self.loss_fn( d_real, real_ones_tsr )
        loss_D_fake = self.loss_fn( d_fake, fake_zeros_tsr )
        loss_D = loss_D_real + loss_D_fake

        return loss_D, loss_D_real, loss_D_fake

    def forward_G(self, d_fake):
        real_ones_tsr =  torch.ones( d_fake.shape ).to(self.device)
        loss_G = self.loss_fn( d_fake, real_ones_tsr )
        return loss_G

    def forward(self, d_real, d_fake, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            loss, _, _ = self.forward_D( d_real, d_fake )
        # Generator 用の loss
        else:
            loss = self.forward_G( d_fake )

        return loss


class LSGANLoss(nn.Module):
    def __init__(self, device):
        super(LSGANLoss, self).__init__()
        self.device = device
        self.loss_fn = nn.MSELoss()       
        return

    def forward_D(self, d_real, d_fake):
        real_ones_tsr = torch.ones( d_real.shape ).to(self.device)
        fake_zeros_tsr = torch.zeros( d_fake.shape ).to(self.device)
        loss_D_real = self.loss_fn( d_real, real_ones_tsr )
        loss_D_fake = self.loss_fn( d_fake, fake_zeros_tsr )
        loss_D = loss_D_real + loss_D_fake
        return loss_D, loss_D_real, loss_D_fake

    def forward_G(self, d_fake):
        real_ones_tsr =  torch.ones( d_fake.shape ).to(self.device)
        loss_G = self.loss_fn( d_fake, real_ones_tsr )
        return loss_G

    def forward(self, d_real, d_fake, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            loss, _, _ = self.forward_D( d_real, d_fake )
        # Generator 用の loss
        else:
            loss = self.forward_G( d_fake )

        return loss


class HingeGANLoss(nn.Module):
    """
    GAN の Hinge loss
        −min(x−1,0)     if D and real
        −min(−x−1,0)    if D and fake
        −x              if G
    """
    def __init__(self, device):
        self.device = device
        super(HingeGANLoss, self).__init__()
        return

    def forward_D(self, d_real, d_fake):
        zeros_tsr =  torch.zeros( d_real.shape ).to(self.device)
        loss_D_real = - torch.mean( torch.min(d_real - 1, zeros_tsr) )
        #loss_D_fake = - torch.mean( torch.min(-d_fake - 1, zeros_tsr) )
        loss_D_fake = - torch.mean( torch.min(-d_real - 1, zeros_tsr) )
        loss_D = loss_D_real + loss_D_fake
        return loss_D, loss_D_real, loss_D_fake

    def forward_G(self, d_fake):
        real_ones_tsr =  torch.ones( d_fake.shape ).to(self.device)
        loss_G = - torch.mean(d_fake)
        return loss_G

    def forward(self, d_real, d_fake, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            loss, _, _ = self.forward_D( d_real, d_fake )
        # Generator 用の loss
        else:
            loss = self.forward_G( d_fake )

        return loss