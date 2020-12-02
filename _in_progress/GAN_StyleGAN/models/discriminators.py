# -*- coding:utf-8 -*-
import os
import numpy as np
import math
from math import ceil

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import models

#==================================
# PGGAN の識別器
#==================================
# Scaled weight - He initialization
# "explicitly scale the weights at runtime"
class ScaleW:
    '''
    Constructor: name - name of attribute to be scaled
    '''
    def __init__(self, name):
        self.name = name
    
    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        
        return weight * math.sqrt(2 / fan_in)
    
    @staticmethod
    def apply(module, name):
        '''
        Apply runtime scaling to specific module
        '''
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)
    
    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)

def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module


class SLinear(nn.Module):
    # Uniformly set the hyperparameters of Linears
    # "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
    # 5/13: Apply scaled weights
    def __init__(self, dim_in, dim_out):
        super().__init__()
        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


class SConv2d(nn.Module):
    # Uniformly set the hyperparameters of Conv2d
    # "We initialize all weights of the convolutional, fully-connected, and affine transform layers using N(0, 1)"
    # 5/13: Apply scaled weights
    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = quick_scale(conv)

    def forward(self, x):
        return self.conv(x)


class ConvBlock(nn.Module):
    '''
    Used to construct progressive discriminator
    '''
    def __init__(self, in_channel, out_channel, size_kernel1, padding1, 
                 size_kernel2 = None, padding2 = None):
        super().__init__()
        
        if size_kernel2 == None:
            size_kernel2 = size_kernel1
        if padding2 == None:
            padding2 = padding1
        
        self.conv = nn.Sequential(
            SConv2d(in_channel, out_channel, size_kernel1, padding=padding1),
            nn.LeakyReLU(0.2),
            SConv2d(out_channel, out_channel, size_kernel2, padding=padding2),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, image):
        # Downsample now proxyed by discriminator
        # result = nn.functional.interpolate(image, scale_factor=0.5, mode="bilinear", align_corners=False)
        # Conv
        result = self.conv(image)
        return result

class ProgressiveDiscriminator( nn.Module ):
    """
    PGGAN の識別器 D [Discriminator] 側のネットワーク構成を記述したモデル。
    """
    def __init__( self, n_rgb = 3, in_dim_latent = 512, image_size_init = 4, image_size_final = 1024 ):
        super( ProgressiveDiscriminator, self ).__init__()
        self.progress_init = int(np.log2(image_size_init)) - 2
        self.progress_final = int(np.log2(image_size_final)) -2
        in_dims = [ in_dim_latent//32, in_dim_latent//16, in_dim_latent//8, in_dim_latent//4, in_dim_latent//2, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent ]
        out_dims = [ in_dim_latent//16, in_dim_latent//8, in_dim_latent//4, in_dim_latent//2, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent ]
        print( "[ProgressiveDiscriminator] in_dims : ", in_dims )
        print( "[ProgressiveDiscriminator] out_dims : ", out_dims )

        #==============================================
        # RGB から 特徴マップ数への変換を行うネットワーク
        #==============================================
        self.fromRGBs = nn.ModuleList()
        for i, in_dim in enumerate(in_dims):
            self.fromRGBs.append( SConv2d( in_channels=n_rgb, out_channels=in_dim, kernel_size=1, stride=1, padding=0 ) )

        #==============================================
        # 
        #==============================================
        self.blocks = nn.ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(in_dims,out_dims)):
            if( i == len(in_dims) - 1 ):
                self.blocks.append( ConvBlock( in_channel=in_dim+1, out_channel=out_dim, size_kernel1=3, padding1=1, size_kernel2 = 4, padding2 = 0 ) )
            else:
                self.blocks.append( ConvBlock( in_channel=in_dim, out_channel=out_dim, size_kernel1=3, padding1=1, size_kernel2 = None, padding2 = None ) )

        self.out_layer = SLinear(out_dims[-1], 1)
        return

    def minibatchstd(self, input):
        # must add 1e-8 in std for stability
        return (input.var(dim=0) + 1e-8).sqrt().mean().view(1, 1, 1, 1)

    def forward(self, input, progress = 0, alpha = 0.0 ):
        for i in range(progress, -1, -1):
            layer_index = self.progress_final - i
            #print( "progress={}, i={}, layer_index={}".format(progress,i,layer_index) )

            # First layer, need to use from_rgb to convert to n_channel data
            if i == progress: 
                output = self.fromRGBs[layer_index](input)

            # Before final layer, do minibatch stddev
            if i == 0:
                output = torch.cat( ( output, self.minibatchstd(output).expand_as(output[:, 0].unsqueeze(1)) ), dim=1 )

            # Conv
            output = self.blocks[layer_index](output)

            # Not the final layer
            if i > 0:
                # Downsample for further usage
                output = nn.functional.interpolate(output, scale_factor=0.5, mode='bilinear', align_corners=False)

                # Alpha set, combine the result of different layers when input
                if i == progress and 0 <= alpha < 1:
                    output_next = self.fromRGBs[layer_index + 1](input)
                    output_next = nn.functional.interpolate(output_next, scale_factor=0.5, mode = 'bilinear', align_corners=False)
                    output = alpha * output + (1 - alpha) * output_next
                    
        # Now, result is [batch, channel(512), 1, 1]
        # Convert it into [batch, channel(512)], so the fully-connetced layer 
        # could process it.
        output = output.squeeze(2).squeeze(2)
        output = self.out_layer(output)
        return output

