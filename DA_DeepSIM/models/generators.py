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


class Pix2PixHDGenerator( nn.Module ):
    def __init__( self, input_nc = 3, output_nc = 3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Pix2PixHDGenerator, self).__init__()        
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
        

class Pix2PixHDAdaINGenerator( nn.Module ):
    def __init__( self, input_nc = 3, output_nc = 3, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, padding_type='reflect', resize_type = "nearest"):
        assert(n_blocks >= 0)
        super(Pix2PixHDAdaINGenerator, self).__init__()        
        self.n_downsampling = n_downsampling

        activation = nn.ReLU(True)
        # encoder
        encoder = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf * mult * 2), activation]

        self.encoder = nn.Sequential(*encoder)

        # bottle_neck / resnet blocks
        bottle_neck = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            bottle_neck += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.bottle_neck = nn.Sequential(*bottle_neck)

        # decoder / upsample
        self.decoder = nn.ModuleDict(
            { "decoder_{}".format(i) : AdaINResBlock(
                n_hin_channles = int(ngf * 2**(n_downsampling - i)), 
                n_hout_channles = int(ngf * 2**(n_downsampling - i) / 2), 
                n_in_channles = input_nc,
                resize_type = resize_type
            ) for i in range(n_downsampling) }
        )

        # output layer
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()
        )
        return

    def forward(self, input):
        output = self.encoder(input)
        output = self.bottle_neck(output)
        #print( "[bottle_neck] output.shape : ", output.shape )
        for i in range(self.n_downsampling):
            output = self.decoder["decoder_{}".format(i)](output, input)
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)
            #print( "[decoder] output.shape : ", output.shape )

        output = self.output_layer(output)
        return output


