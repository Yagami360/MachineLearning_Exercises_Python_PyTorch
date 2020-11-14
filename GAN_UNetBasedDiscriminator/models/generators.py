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
        

#====================================
# UNet
#====================================
class UNetGenerator(nn.Module):
    """
    任意の upsampling / downsampling 層数での UNet 生成器
    """
    def __init__(self, n_in_channels=3, n_out_channels=3, n_fmaps=64, n_downsampling=4, norm_type='batch'):
        super( UNetGenerator, self ).__init__()
        self.n_downsampling = n_downsampling
        if norm_type == 'batch':
            self.norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            self.norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        else:
            raise NotImplementedError()

        def conv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                self.norm_layer( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True ),
                nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                self.norm_layer( out_dim ),
            )
            return model

        def dconv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.ConvTranspose2d( in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1 ),
                self.norm_layer(out_dim),
                nn.LeakyReLU( 0.2, inplace=True ),
            )
            return model
        
        # encoder
        in_dim = n_in_channels
        out_dim = n_fmaps
        self.encoder = nn.ModuleDict()
        for i in range(n_downsampling):
            self.encoder["conv_{}".format(i+1)] = conv_block( in_dim, out_dim )
            self.encoder["pool_{}".format(i+1)] = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
            in_dim = n_fmaps * (2**(i))
            out_dim = n_fmaps * (2**(i+1))

        # bottle neck
        self.bridge = conv_block( n_fmaps * (2**(n_downsampling-1)), n_fmaps*(2**(n_downsampling-1))*2 )

        # decoder
        self.decoder = nn.ModuleDict()
        for i in range(n_downsampling):
            in_dim = n_fmaps * (2**(n_downsampling-i))
            out_dim = int( in_dim / 2 )
            self.decoder["deconv_{}".format(i+1)] = dconv_block( in_dim, out_dim )
            self.decoder["conv_{}".format(i+1)] = conv_block( in_dim, out_dim )

        # 出力層
        self.out_layer = nn.Sequential( nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ), nn.Tanh() )
        return

    def forward(self, input):
        output = input

        skip_connections = []
        for i in range(self.n_downsampling):
            output = self.encoder["conv_{}".format(i+1)](output)
            skip_connections.append( output.clone() )
            output = self.encoder["pool_{}".format(i+1)](output)
            #print("[UNetGenerator] encoder_{} / output.shape={}".format(i+1, output.shape) )

        output = self.bridge(output)
        #print("[UNetGenerator] bridge / output.shape={}".format(i+1, output.shape) )

        for i in range(self.n_downsampling):
            output = self.decoder["deconv_{}".format(i+1)](output)
            output = self.decoder["conv_{}".format(i+1)]( torch.cat( [output, skip_connections[-1 - i]], dim=1 ) )
            #print("[UNetGenerator] decoder_{} / output.shape={}".format(i+1, output.shape) )

        output = self.out_layer(output)
        #print("[UNetGenerator] out_layer / output.shape : ", output.shape )
        return output