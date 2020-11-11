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

        # 識別器のネットワークでは、Patch GAN を採用するが、
        # patchを切り出したり、ストライドするような処理は、直接的には行わない。その代りに、これを畳み込みで表現する。
        # つまり、CNNを畳み込んで得られる特徴マップのある1pixelは、入力画像のある領域(Receptive field)の影響を受けた値になるが、
        # 裏を返せば、ある1pixelに影響を与えられるのは、入力画像のある領域だけ。
        # そのため、「最終出力をあるサイズをもった特徴マップにして、各pixelにて真偽判定をする」ことと 、「入力画像をpatchにして、各patchの出力で真偽判定をする」ということが等価になるためである。
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


class SinGANPatchGANDiscriminator(nn.Module):
    def __init__( 
        self, 
        input_nc=3, output_nc=3,
        n_fmaps=32, n_layers=5, kernel_size=3, stride=1, padding=0,
        norm_type='batch',
    ):
        super(SinGANPatchGANDiscriminator, self).__init__()

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
        )
        return        

    def forward(self, image_gen ):
        output = self.head_layer(image_gen)
        #print( "[head_layer] output.shape : ", output.shape )
        output = self.body_layer(output)
        #print( "[body_layer] output.shape : ", output.shape )
        output = self.output_layer(output)
        #print( "[output_layer] output.shape : ", output.shape )
        return output


class SinGANDiscriminator(nn.Module):
    def __init__( 
        self, 
        input_nc=3, output_nc=3,
        n_fmaps=32, n_layers=5, kernel_size=3, stride=1, padding=0,
        norm_type='batch',
        train_progress_init = 0, train_progress_max = 8, 
    ):
        super(SinGANDiscriminator, self).__init__()
        self.train_progress_init = train_progress_init
        self.train_progress_max = train_progress_max

        self.discriminators = nn.ModuleDict(
            { "discriminator_{}".format(i) : 
                SinGANPatchGANDiscriminator( input_nc, output_nc, n_fmaps=n_fmaps, n_layers=n_layers, kernel_size=kernel_size, stride=stride, padding=padding, )
                for i in range(train_progress_max)
            }
        )

    def freeze_grads( self, train_progress ):
        for param in self.discriminators["discriminator_{}".format(train_progress)].parameters():
            param.requires_grad_(False)
        return

    def forward(self, image_gen, train_progress = 0):
        output = self.discriminators["discriminator_{}".format(train_progress)](image_gen)
        return output
