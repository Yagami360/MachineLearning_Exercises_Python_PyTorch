# -*- coding:utf-8 -*-
import os
import numpy as np
import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torchvision

from models.network_base import Reshape, GLU, Swish

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
        in_dims_dict = { "4x4" : n_fmaps*4*4*2, "8x8" : n_fmaps*4*4, "16x16": n_fmaps*4*4//2, "32x32": n_fmaps*4*4//4, "64x64": n_fmaps*4*4//8, "128x128": n_fmaps*4*4//8, "256x256": n_fmaps*4*4//16, "512x512": n_fmaps*4*4//32, "1024x1024": n_fmaps*4*4//64 }
        out_dims_dict = { "4x4" : n_fmaps*4*4, "8x8" : n_fmaps*4*4//2, "16x16": n_fmaps*4*4//4, "32x32": n_fmaps*4*4//8, "64x64": n_fmaps*4*4//8, "128x128": n_fmaps*4*4//16, "256x256": n_fmaps*4*4//32, "512x512": n_fmaps*4*4//64, "1024x1024": n_fmaps*4*4//128 }
        print( "in_dims_dict : ", in_dims_dict )
        print( "out_dims_dict : ", out_dims_dict )

        # 入力層
        self.input_layer = InputLayer(z_dims, out_dims_dict["4x4"] )

        # アップサンプリング層
        self.upsampling_8 = UpsamplingWithNoizeLayer(in_dims_dict["8x8"], out_dims_dict["8x8"])
        self.upsampling_16 = UpsamplingLayer(in_dims_dict["16x16"], out_dims_dict["16x16"])
        self.upsampling_32 = UpsamplingWithNoizeLayer(in_dims_dict["32x32"], out_dims_dict["32x32"])
        self.upsampling_64 = UpsamplingLayer(in_dims_dict["64x64"], out_dims_dict["64x64"])
        self.upsampling_128 = UpsamplingWithNoizeLayer(in_dims_dict["128x128"], out_dims_dict["128x128"])
        if( image_size >= 256 ):
            self.upsampling_256 = UpsamplingLayer(in_dims_dict["256x256"], out_dims_dict["256x256"])
        if( image_size >= 512 ):
            self.upsampling_512 = UpsamplingWithNoizeLayer(in_dims_dict["512x512"], out_dims_dict["512x512"])
        if( image_size >= 1024 ):
            self.upsampling_1024 = UpsamplingLayer(in_dims_dict["1024x1024"], out_dims_dict["1024x1024"])

        # SLE 
        self.sle_8 = SLEblock(out_dims_dict["8x8"],out_dims_dict["128x128"])
        if( image_size > 256 ):
            self.sle_16 = SLEblock(out_dims_dict["16x16"],out_dims_dict["256x256"])
        if( image_size > 512 ):
            self.sle_32 = SLEblock(out_dims_dict["32x32"],out_dims_dict["512x512"])
        if( image_size > 1024 ):
            self.sle_64 = SLEblock(out_dims_dict["64x64"],out_dims_dict["1024x1024"])

        # 出力層
        if( image_size == 256 ):
            self.output_layer = OutputLayer(in_dims_resmax=out_dims_dict["256x256"], in_dims_res256 = out_dims_dict["128x128"], out_dims = out_dims)
        if( image_size == 512 ):
            self.output_layer = OutputLayer(in_dims_resmax=out_dims_dict["512x512"], in_dims_res256 = out_dims_dict["128x128"], out_dims = out_dims)
        if( image_size == 1024 ):
            self.output_layer = OutputLayer(in_dims_resmax=out_dims_dict["1024x1024"], in_dims_res256 = out_dims_dict["128x128"], out_dims = out_dims)

        return

    def forward(self, latent_z):
        feat4 = self.input_layer(latent_z)
        #print( "feat4.shape : ", feat4.shape )
        feat8 = self.upsampling_8(feat4)
        #print( "feat8.shape : ", feat8.shape )
        feat16 = self.upsampling_16(feat8)
        #print( "feat16.shape : ", feat16.shape )
        feat32 = self.upsampling_32(feat16)
        #print( "feat32.shape : ", feat32.shape )
        feat64 = self.upsampling_64(feat32)
        #print( "feat64.shape : ", feat64.shape )
        feat128 = self.upsampling_128(feat64)
        #print( "feat128.shape : ", feat128.shape )
        feat128_skip = self.sle_8(feat8, feat128)
        #print( "feat128_skip.shape : ", feat128_skip.shape )

        feat256 = self.upsampling_256(feat128_skip)
        #print( "feat256.shape : ", feat256.shape )
        if( self.image_size == 256 ):
            output_resmax, output_res128 = self.output_layer(feat256, feat128_skip)
            return output_resmax, output_res128

        if( self.image_size == 512 ):
            feat256_skip = self.sle_16(feat16, feat256)
            feat512 = self.upsampling_512(feat256_skip)
            output_resmax, output_res128 = self.output_layer(feat512, feat128_skip)
            return output_resmax, output_res128

        if( self.image_size == 1024 ):
            feat256_skip = self.sle_16(feat16, feat256)
            feat512 = self.upsampling_512(feat256_skip)
            feat512_skip = self.sle_32(feat32, feat512)
            feat1024 = self.upsampling_1024(feat512_skip)
            output_resmax, output_res128 = self.output_layer(feat1024, feat128_skip)
            return output_resmax, output_res128

