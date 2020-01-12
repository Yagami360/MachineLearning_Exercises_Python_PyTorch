#coding=utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import Vgg19

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
    def __init__(self, device, w_sigmoid_D = True ):
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


#============================================
# PixPix-HD loss
#============================================
class FeatureMatchingLoss(nn.Module):
    """
    Pix2Pix-HD の Feature Matching Loss
    """
    def __init__(self, device, n_dis = 3, n_layers_D = 3):
        super(FeatureMatchingLoss, self).__init__()
        self.device = device
        self.n_dis = n_dis
        self.n_layers_D = n_layers_D
        self.loss_fn = torch.nn.L1Loss()
        return

    def forward(self, d_reals, d_fakes ):
        loss = 0
        weights_feat = 4.0 / (self.n_layers_D + 1)
        weights_D = 1.0 / self.n_dis
        for i in range(self.n_dis):
            for j in range(len(d_fakes[i])-1):
                loss += weights_D * weights_feat * self.loss_fn(d_fakes[i][j], d_reals[i][j].detach())

        return loss


class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()        
        self.device = device
        self.vgg = Vgg19().to(self.device)
        self.loss_fn = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        #print( "x_vgg[0].shape : ", x_vgg[0].shape )
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.loss_fn(x_vgg[i], y_vgg[i].detach())        
        return loss
