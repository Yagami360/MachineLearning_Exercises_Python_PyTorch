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


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Pix2PixHD( nn.Module ):
    def __init__( self, input_nc = 3, output_nc = 3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Pix2PixHD, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)        
        

class AdaIN(nn.Module):
    def __init__( self, n_in_channles = 3, n_out_channles = 3 , net_type = "conv" ):
        super(AdaIN, self).__init__()
        self.net_type = net_type
        self.batch_norm = nn.BatchNorm2d(n_in_channles)
        if( self.net_type == "fc" ):
            self.gamma = nn.Linear(n_in_channles, n_out_channles)
            self.beta = nn.Linear(n_in_channles, n_out_channles)
        elif( self.net_type == "conv" ):
            self.gamma = nn.Conv2d(n_in_channles, n_out_channles, kernel_size=3, stride=1, padding=1)
            self.beta = nn.Conv2d(n_in_channles, n_out_channles, kernel_size=3, stride=1, padding=1)
        else:
            NotImplementedError()

        return

    def forward(self, input):
        h_act = self.batch_norm(input)

        if( self.net_type == "fc" ):
            input_fc = torch.flatten(input)
            gamma = self.gamma(input_fc)
            beta = self.beta(input_fc)
            gamma = gamma.view(input.shape)
            beta = beta.view(input.shape)

        elif( self.net_type == "conv" ):
            gamma = self.gamma(input)
            beta = self.beta(input)

        print( "gamma.shape", gamma.shape )     # torch.Size([B, 3, 128, 128]) 
        print( "beta.shape", beta.shape )       # torch.Size([B, 3, 128, 128]) 
        h_after = gamma * h_act + beta
        print( "h_after.shape", h_after.shape )
        return h_after

class Pix2PixHDAdaIN( nn.Module ):
    def __init__( self, input_nc = 3, output_nc = 3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Pix2PixHDAdaIN, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)

        # AdaIN
        self.adain = AdaIN( n_in_channles = 3, n_out_channles = 3 , net_type = "conv" )
        return

    def forward(self, input):
        h_after = self.adain(input)
        output = self.model(h_after)
        return output