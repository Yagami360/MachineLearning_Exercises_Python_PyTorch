#coding=utf-8
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class HingeGANLoss(nn.Module):
    """
    GAN の Hinge loss
        −min(x−1,0)     if D and real
        −min(−x−1,0)    if D and fake
        −x              if G
    """
    def __init__(self, device):
        super(HingeGANLoss, self).__init__()
        self.device = device
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

    """
    def forward(self, d_input, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            zeros_tsr =  torch.zeros( d_input.shape ).to(self.device)
            loss_real = - torch.mean( torch.min(d_input - 1, zeros_tsr) )
            loss_fake = - torch.mean( torch.min(-d_input - 1, zeros_tsr) )
            loss = loss_real + loss_fake

        # Generator 用の loss
        else:
            loss = - torch.mean(d_input)

        return loss
    """