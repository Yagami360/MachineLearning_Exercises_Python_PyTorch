# -*- coding:utf-8 -*-
import os
import numpy as np

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
    def __init__( self, in_dim = 3, n_fmaps = 64 ):
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

        #self.layer1 = discriminator_block1( in_dim * 2, n_fmaps )
        self.layer1 = discriminator_block1( in_dim, n_fmaps )
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
        in_dim = 3, n_fmaps = 64,
        n_dis = 3,                # 識別器の数
#       n_layers = 3,        
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
            setattr( self, 'scale'+str(i)+'_layer0', discriminator_block1( in_dim, n_fmaps, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer1', discriminator_block2( n_fmaps, n_fmaps*2, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer2', discriminator_block2( n_fmaps*2, n_fmaps*4, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer3', discriminator_block2( n_fmaps*4, n_fmaps*8, 1, 2) )
            setattr( self, 'scale'+str(i)+'_layer4', discriminator_block3( n_fmaps*8, 1, 1, 2) )

        """
        # この方法だと、各 Sequential ブロックに名前をつけられない（連番になる）
        self.layers = nn.ModuleList()
        for i in range(self.n_dis):
            self.layers.append( discriminator_block1( in_dim*2, n_fmaps, 2, 2) )
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
# light-weight GAN の 識別器
#------------------------------------
class InputLayer(nn.Module):
    def __init__(self, in_dims, n_fmaps, image_size = 1024 ):
        super(InputLayer, self).__init__()
        if( image_size == 256 ):
            self.layers = nn.Sequential(
                nn.Conv2d(in_dims, n_fmaps//4, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif( image_size == 512 ):
            self.layers = nn.Sequential(
                nn.Conv2d(in_dims, n_fmaps//4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(n_fmaps//4, n_fmaps//4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(n_fmaps//4),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif( image_size == 1024 ):
            self.layers = nn.Sequential(
                nn.Conv2d(in_dims, n_fmaps//8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(n_fmaps//8, n_fmaps//8, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(n_fmaps//8),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            NotImplementedError()

        return

    def forward(self, input):
        output = self.layers(input)
        return output

class DownBlock(nn.Module):
    def __init__(self, in_dims, out_dims, h = 2, w = 2 ):
        super(DownBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.AvgPool2d((h,w)),
            nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return

    def forward(self, input):
        output1 = self.layer1(input)
        output2 = self.layer2(input)
        output = output1 + output2
        return output

class SLEblock(nn.Module):
    """
    light-weight GAN の SLE [Skip-Layer Excitation module]
    """
    def __init__(self, in_dims, out_dims, h = 4, w = 4 ):
        super(SLEblock, self).__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((h,w)),
            nn.Conv2d(in_dims, out_dims, kernel_size=4, stride=1, padding=0, bias=False),            
            Swish(),
            nn.Conv2d(out_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False),                        
            nn.Sigmoid(),
        )
        return
        
    def forward(self, input, input_skip ):
        #print( "[SLEblock] input.shape : ", input.shape )
        #print( "[SLEblock] input_skip.shape : ", input_skip.shape )
        output = self.layers(input)
        #print( "[SLEblock] output.shape : ", output.shape )
        output = output * input_skip
        #print( "[SLEblock] output.shape : ", output.shape )
        return output

class LightweightGANDiscriminator(nn.Module):
    """
    light-weight GAN の識別器
    """
    def __init__( self, in_dim = 3, n_fmaps = 64, image_size = 1024 ):
        super(LightweightGANDiscriminator, self).__init__()
        self.input_layer = InputLayer(in_dim, n_fmaps, image_size )
        self.down = DownBlock(n_fmaps//4, n_fmaps//2)
        return

    def forward(self, input, input_128 = None, interpolate_mode = "bilinear" ):
        if input_128 is None : 
            input_128 = F.interpolate(input, size=128, mode = interpolate_mode)

        output = self.input_layer(input)
        print( "[LightweightGANDiscriminator] output.shape", output.shape )
        output = self.down(output)
        print( "[LightweightGANDiscriminator] output.shape", output.shape )

        return output
