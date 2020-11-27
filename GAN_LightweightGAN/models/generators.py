# -*- coding:utf-8 -*-
import os
import numpy as np
import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torchvision

from models.network_base import Reshape, GLU

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        return

    def forward(self, input, noise=None):
        self.weight.to(input.device)
        if noise is None:
            batch, _, height, width = input.shape
            noise = torch.randn(batch, 1, height, width).to(input.device)

        return input + self.weight * noise

class InputLayer(nn.Module):
    def __init__(self, z_dims, out_dims ):
        super(InputLayer, self).__init__()
        self.layers = nn.Sequential(
            Reshape(1,1),
            spectral_norm( nn.ConvTranspose2d(z_dims, out_dims*2, kernel_size=4, stride=1, padding=0, bias=False) ),
            nn.BatchNorm2d(out_dims*2),
            GLU(split_dim=1),
        )
        return

    def forward(self, latent_z):
        output = self.layers(latent_z)
        return output

class UpsamplingLayer(nn.Module):
    def __init__(self, in_dims, out_dims, scale_factor = 2, mode = "nearest"):
        super(UpsamplingLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample( scale_factor = scale_factor, mode = mode ),
            nn.Conv2d(in_dims, out_dims*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dims*2),
            GLU(split_dim=1),
        )
        return

    def forward(self, input):
        output = self.layers(input)
        return output

class UpsamplingWithNoizeLayer(nn.Module):
    def __init__(self, in_dims, out_dims, scale_factor = 2, mode = "nearest"):
        super(UpsamplingWithNoizeLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample( scale_factor = scale_factor, mode = mode ),
            nn.Conv2d(in_dims, out_dims*2, kernel_size=3, stride=1, padding=1, bias=False),
            NoiseInjection(),
            nn.BatchNorm2d(out_dims*2),
            GLU(split_dim=1),
            nn.Conv2d(out_dims, out_dims*2, kernel_size=3, stride=1, padding=1, bias=False),
            NoiseInjection(),
            nn.BatchNorm2d(out_dims*2),
            GLU(split_dim=1),
        )
        return

    def forward(self, input):
        output = self.layers(input)
        return output

class OutputLayer(nn.Module):
    def __init__(self, in_dims_resmax, in_dims_res256, out_dims = 3 ):
        super(OutputLayer, self).__init__()
        # 最大解像度での出力画像
        self.to_resmax = nn.Sequential(
            spectral_norm( nn.Conv2d( in_dims_resmax, out_dims, kernel_size=3, stride=1, padding=1) ),
            nn.Tanh(),
        )

        # 128 解像度での出力画像（識別器に入力するための出力画像）
        self.to_res128 = nn.Sequential(
            spectral_norm( nn.Conv2d( in_dims_res256, out_dims, kernel_size=1, stride=1, padding=0) ),
            nn.Tanh(),
        )
        return

    def forward(self, input_resmax, input_res128 ):
        output_resmax = self.to_resmax(input_resmax)
        output_res128 = self.to_res128(input_res128)
        return output_resmax, output_res128

class LightweightGANGenerator(nn.Module):
    """
    light-weight GAN の生成器
    """
    def __init__(self, z_dims = 256, n_fmaps = 64, out_dims = 3, image_size = 1024 ):
        super(LightweightGANGenerator, self).__init__()
        self.image_size = image_size
        self.n_upsamplings = int(np.log2(image_size)) - 2
        self.size_to_idx = {
            str(image_size//2**i) : self.n_upsamplings - i for i in range(self.n_upsamplings+1)
        }
        print( "self.size_to_idx", self.size_to_idx )

        # 入力層
        self.input_layer = InputLayer(z_dims, n_fmaps*4*4)

        # アップサンプリング層
        in_fmap_dims = n_fmaps*4*4
        self.upsampling_layers = nn.ModuleDict(
            { "upsampling_{}".format(i+1) : 
                UpsamplingLayer(in_fmap_dims//(2**i), in_fmap_dims//(2**(i+1)))
                for i in range(self.n_upsamplings)
            }
        )

        # SLE 
        pass

        # 出力層
        self.output_layer = OutputLayer(in_dims_resmax=in_fmap_dims//(2**(self.n_upsamplings)), in_dims_res256 = n_fmaps*4*4//16, out_dims = out_dims)
        return

    def forward(self, latent_z):
        output_list = []
        output = self.input_layer(latent_z)
        output_list.append(output)
        print( "[LightweightGANGenerator] output.shape", output.shape )

        for i in range(self.n_upsamplings):
            output = self.upsampling_layers["upsampling_{}".format(i+1)](output_list[-1])
            print( "[LightweightGANGenerator] output.shape", output.shape )
            output_list.append(output)

        # self.size_to_idx {'1024': 8, '512': 7, '256': 6, '128': 5, '64': 4, '32': 3, '16': 2, '8': 1, '4': 0}
        output_resmax, output_res128 = self.output_layer(output_list[-1], output_list[self.size_to_idx[str(128)]] )
        print( "[LightweightGANGenerator] output_resmax.shape", output_resmax.shape )
        print( "[LightweightGANGenerator] output_res128.shape", output_res128.shape )
        return [output_resmax, output_res128]
