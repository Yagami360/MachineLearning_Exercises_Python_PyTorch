# -*- coding:utf-8 -*-
import os
import numpy as np
from math import ceil

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import models

#==================================
# StyleGAN の識別器
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


class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    PGGAN が提案している equalized learning rate の手法（学習安定化のための手法）に従って、
    前の層（preceding layer）の重みを正則化する。
    1. 生成器と識別器のネットワークの各層（i）の重み w_i  の初期値を、w_i~N(0,1)  で初期化。
    2. 初期化した重みを、各層の実行時（＝順伝搬での計算時）に、以下の式で再設定する。
        w^^ = w_i/c  (標準化定数 c=\sqrt(2/層の数))
    """
    def __init__(self, pre_layer):
        """
        [Args]
            pre_layer : <nn.Module> 重みの正規化を行う層
        """
        super(WScaleLayer, self).__init__()
        self._pre_layer = pre_layer
        self._scale = (torch.mean(self._pre_layer.weight.data ** 2)) ** 0.5            # 標準化定数 c
        self._pre_layer.weight.data.copy_(self._pre_layer.weight.data / self._scale)     # w^ = w_i/c
        self._bias = None
        if self._pre_layer.bias is not None:
            self._bias = self._pre_layer.bias
            self._pre_layer.bias = None

    def forward(self, x):
        x = self._scale * x
        if self._bias is not None:
            x += self._bias.view(1, self._bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        param_str = '(pre_layer = %s)' % (self._pre_layer.__class__.__name__)
        return self.__class__.__name__ + param_str


class ProgressiveDiscriminator( nn.Module ):
    """
    PGGAN の識別器 D [Discriminator] 側のネットワーク構成を記述したモデル。
    """
    def __init__(
        self,
        init_image_size = 4,
        final_image_size = 32,
        n_fmaps = 128,
        n_rgb = 3,
    ):
        super( ProgressiveDiscriminator, self ).__init__()

        #==============================================
        # RGB から 特徴マップ数への変換を行うネットワーク
        #==============================================
        self.fromRGBs = nn.ModuleList()

        # 4 × 4
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers = nn.Sequential( *layers )
        self.fromRGBs.append( layers )

        # 8 × 8
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers = nn.Sequential( *layers )
        self.fromRGBs.append( layers )

        # 16 × 16
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers = nn.Sequential( *layers )
        self.fromRGBs.append( layers )

        # 32 × 32
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers = nn.Sequential( *layers )
        self.fromRGBs.append( layers )

        #print( "fromRGBs :", self.fromRGBs )

        #==============================================
        # 0.0 < α <= 1.0 での conv 層
        #==============================================
        self.blocks = nn.ModuleList()

        #-----------------------------------------
        # 4 × 4
        #-----------------------------------------
        layers = []

        # conv 3 × 3 : shape = [n_fmaps, 4, 4] → [n_fmaps, 4, 4]
        layers.append( nn.Conv2d( in_channels=n_fmaps+1, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )

        # conv 4 × 4 : shape = [n_fmaps, 4, 4] → [n_fmaps, 1, 1]
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=4, stride=1, padding=0 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )

        # conv 1 × 1 : shape = [n_fmaps, 1, 1] → [1, 1, 1]
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=1, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.Sigmoid() )

        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #-----------------------------------------
        # 8 × 8
        #-----------------------------------------
        layers = []

        # conv 3 × 3 : [n_fmaps, 8, 8] → []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )

        # conv 3 × 3 : [n_fmaps, 8, 8] → []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #-----------------------------------------
        # 16 × 16
        #-----------------------------------------
        layers = []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #-----------------------------------------
        # 32 × 32
        #-----------------------------------------
        layers = []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #print( "blocks :", blocks )

        return

    def minibatchstd(self, input):
        # must add 1e-8 in std for stability
        return (input.var(dim=0) + 1e-8).sqrt().mean().view(1, 1, 1, 1)

    def forward(self, input, train_progress=0 ):
        """
        [Args]
            input : <Tensor> ネットワークへの入力
            train_progress : <float> 現在の Training Progress / 0.0 → 0.0 ~ 1.0 → 1.0 → 1.0 ~ 2.0 → 2.0 → ...
        """
        #-----------------------------------------
        # 学習開始時点（α=0.0）
        #-----------------------------------------
        if( train_progress % 1 == 0 ):
            # shape = [1, x, x] → [n_fmaps, x, x]            
            output = self.fromRGBs[int(ceil(train_progress))](input)

            # shape = [n_fmaps, x, x] → [n_fmaps, 4, 4]
            for i in range(int(train_progress), 0, -1):
                output = self.blocks[i](output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2)  # Downsampling

            # shape = [n_fmaps, 4, 4] → [n_fmaps+1, 4, 4]
            output = torch.cat( ( output, self.minibatchstd(output).expand_as(output[:, 0].unsqueeze(1)) ), dim=1 )   # tmp : torch.Size([16, 129, 4, 4])

            # shape = [n_fmaps, 4, 4] → [1, 1, 1]
            output = self.blocks[0]( output )
            output = output.squeeze()

        #-----------------------------------------
        # 0.0 < α <= 1.0
        #-----------------------------------------
        else:
            alpha = train_progress - int(train_progress)

            output0 = F.avg_pool2d(input, kernel_size=2, stride=2)  # Downsampling
            output0 = self.fromRGBs[int(train_progress)](output0)

            output1 = self.fromRGBs[int(ceil(train_progress))](input)
            output1 = self.blocks[int(ceil(train_progress))](output1)
            output1 = F.avg_pool2d(output1, kernel_size=2, stride=2)  # Downsampling

            output = alpha * output1 + (1 - alpha) * output0

            # shape = [n_fmaps, x, x] → [n_fmaps, 4, 4]
            for i in range(int(train_progress), 0, -1):
                output = self.blocks[i](output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2)  # Downsampling

            # shape = [n_fmaps, 4, 4] → [n_fmaps+1, 4, 4]
            output = torch.cat( ( output, self.minibatchstd(output).expand_as(output[:, 0].unsqueeze(1)) ), dim=1 )   # tmp : torch.Size([16, 129, 4, 4])

            # shape = [n_fmaps, 4, 4] → [1, 1, 1]
            output = self.blocks[0]( output )
            output = output.squeeze()

        return output