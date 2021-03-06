# -*- coding:utf-8 -*-
import os
import numpy as np
import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import models

#====================================
# 識別器
#====================================
class PatchGANDiscriminator( nn.Module ):
    """
    PatchGAN の識別器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_fmaps = 32
    ):
        super( PatchGANDiscriminator, self ).__init__()

        def discriminator_block1( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        def discriminator_block2( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.InstanceNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        #self.layer1 = discriminator_block1( n_in_channels * 2, n_fmaps )
        self.layer1 = discriminator_block1( n_in_channels, n_fmaps )
        self.layer2 = discriminator_block2( n_fmaps, n_fmaps*2 )
        self.layer3 = discriminator_block2( n_fmaps*2, n_fmaps*4 )
        self.layer4 = discriminator_block2( n_fmaps*4, n_fmaps*8 )

        self.output_layer = nn.Sequential(
            nn.ZeroPad2d( (1, 0, 1, 0) ),
            nn.Conv2d( n_fmaps*8, 1, 4, padding=1, bias=False )
        )

    def forward(self, input ):
        #output = torch.cat( [x, y], dim=1 )
        output = self.layer1( input )
        output = self.layer2( output )
        output = self.layer3( output )
        output = self.layer4( output )
        output = self.output_layer( output )
        output = output.view(-1)
        return output


class MultiscaleDiscriminator(nn.Module):
    """
    Pix2Pix-HD のマルチスケール識別器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_fmaps = 64,
        n_dis = 3,                # 識別器の数
#        n_layers = 3,        
    ):
        super( MultiscaleDiscriminator, self ).__init__()
        self.n_dis = n_dis
        #self.n_layers = n_layers
        
        def discriminator_block1( in_dim, out_dim, stride, padding ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride, padding ),
                nn.LeakyReLU( 0.2, inplace=True ),
            )
            return model

        def discriminator_block2( in_dim, out_dim, stride, padding ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride, padding ),
                nn.InstanceNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        def discriminator_block3( in_dim, out_dim, stride, padding ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride, padding ),
            )
            return model

        # マルチスケール識別器で、入力画像を 1/2 スケールにする層
        self.downsample_layer = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        # setattr() を用いて self オブジェクトを動的に生成することで、各 Sequential ブロックに名前をつける
        for i in range(self.n_dis):
            setattr( self, 'scale'+str(i)+'_layer0', discriminator_block1( n_in_channels, n_fmaps, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer1', discriminator_block2( n_fmaps, n_fmaps*2, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer2', discriminator_block2( n_fmaps*2, n_fmaps*4, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer3', discriminator_block2( n_fmaps*4, n_fmaps*8, 1, 2) )
            setattr( self, 'scale'+str(i)+'_layer4', discriminator_block3( n_fmaps*8, 1, 1, 2) )

        """
        # この方法だと、各 Sequential ブロックに名前をつけられない（連番になる）
        self.layers = nn.ModuleList()
        for i in range(self.n_dis):
            self.layers.append( discriminator_block1( n_in_channels*2, n_fmaps, 2, 2) )
            self.layers.append( discriminator_block2( n_fmaps, n_fmaps*2, 2, 2) )
            self.layers.append( scdiscriminator_block2( n_fmaps*2, n_fmaps*4, 2, 2)ale_layer )
            self.layers.append( discriminator_block2( n_fmaps*4, n_fmaps*8, 1, 2) )
            self.layers.append( discriminator_block3( n_fmaps*8, 1, 1, 2) )
        """
        return

    def forward(self, input ):
        """
        [Args]
            input : 入力画像 <torch.Float32> shape =[N,C,H,W]
        [Returns]
            outputs_allD : shape=[n_dis, n_layers=5, tensor=[N,C,H,W] ]
        """
        #input = torch.cat( [x, y], dim=1 )

        outputs_allD = []
        for i in range(self.n_dis):
            if i > 0:
                # 入力画像を 1/2 スケールにする
                input = self.downsample_layer(input)

            scale_layer0 = getattr( self, 'scale'+str(i)+'_layer0' )
            scale_layer1 = getattr( self, 'scale'+str(i)+'_layer1' )
            scale_layer2 = getattr( self, 'scale'+str(i)+'_layer2' )
            scale_layer3 = getattr( self, 'scale'+str(i)+'_layer3' )
            scale_layer4 = getattr( self, 'scale'+str(i)+'_layer4' )

            outputs_oneD = []
            outputs_oneD.append( scale_layer0(input) )
            outputs_oneD.append( scale_layer1(outputs_oneD[-1]) )
            outputs_oneD.append( scale_layer2(outputs_oneD[-1]) )
            outputs_oneD.append( scale_layer3(outputs_oneD[-1]) )
            outputs_oneD.append( scale_layer4(outputs_oneD[-1]) )
            outputs_allD.append( outputs_oneD )

        return outputs_allD


#------------------------------------
# A U-Net Based Discriminator
#------------------------------------
class UNetDiscriminator(nn.Module):
    """
    U-Net Based Discriminator
    """
    def __init__(self, n_in_channels=3, n_out_channels=1, n_fmaps=64, n_downsampling=4, norm_type='batch'):
        super( UNetDiscriminator, self ).__init__()
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
        self.out_layer = nn.Sequential( nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),)
        return

    def forward(self, input):
        #print("[UNetDiscriminator] input.shape : ", input.shape )
        encode = input

        skip_connections = []
        for i in range(self.n_downsampling):
            encode = self.encoder["conv_{}".format(i+1)](encode)
            skip_connections.append( encode.clone() )
            encode = self.encoder["pool_{}".format(i+1)](encode)
            #print("[UNetDiscriminator] encoder_{} / output.shape={}".format(i+1, output.shape) )

        encode = self.bridge(encode)
        #print("[UNetDiscriminator] bridge / output.shape={}".format(i+1, output.shape) )

        decode = encode.clone()
        for i in range(self.n_downsampling):
            decode = self.decoder["deconv_{}".format(i+1)](decode)
            decode = self.decoder["conv_{}".format(i+1)]( torch.cat( [decode, skip_connections[-1 - i]], dim=1 ) )
            #print("[UNetDiscriminator] decoder_{} / output.shape={}".format(i+1, output.shape) )

        decode = self.out_layer(decode)
        #print("[UNetDiscriminator] out_layer / output.shape : ", output.shape )
        return encode, decode

