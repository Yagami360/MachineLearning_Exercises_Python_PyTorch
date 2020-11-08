# -*- coding:utf-8 -*-
import os
import numpy as np
import functools

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
    def __init__( self, input_nc = 3, output_nc = 3, ngf=64, n_downsampling=3, n_blocks=9, norm_type = 'batch', padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Pix2PixHDGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            raise NotImplementedError()

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
        

class PatchGANGenerator(nn.Module):
    def __init__( 
        self, 
        input_nc=3, output_nc=3,
        n_fmaps=32, n_layers=5, kernel_size=3, stride=1, padding=0,
        norm_type='batch',
    ):
        super(PatchGANGenerator, self).__init__()

        def define_conv_block( in_dim, out_dim, norm_layer, kernel_size=4, stride=2, padding=1 ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size, stride=stride, padding=padding ),
                norm_layer( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            raise NotImplementedError()

        # １段目の層
        self.head_layer = define_conv_block( in_dim = input_nc, out_dim = n_fmaps, norm_layer = norm_layer, kernel_size=kernel_size, stride=stride, padding=padding )

        # 中間層
        self.body_layer = nn.Sequential()
        for i in range(n_layers-2):
            n_fmaps_mult = int(n_fmaps/pow(2,(i+1)))
            #print( "n_fmaps_mult : ", n_fmaps_mult )
            conv_block = define_conv_block(
                in_dim = max(n_fmaps_mult * 2, n_fmaps), out_dim = max(n_fmaps_mult, n_fmaps),
                norm_layer = norm_layer, kernel_size=kernel_size, stride=stride, padding=padding 
            )
            self.body_layer.add_module( "conv_block_{}".format(i+1), conv_block )

        # 出力層        
        self.output_layer = nn.Sequential(
            nn.Conv2d(max(n_fmaps_mult, n_fmaps), output_nc, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Tanh()
        )
        return        

    def forward(self, noize_image_z):
        output = self.head_layer(noize_image_z)
        output = self.body_layer(output)
        output = self.output_layer(output)
        return output