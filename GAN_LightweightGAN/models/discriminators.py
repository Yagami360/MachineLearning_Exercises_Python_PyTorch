# -*- coding:utf-8 -*-
import os
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torchvision
from torchvision import models

from models.network_base import GLU, Swish

#====================================
# 識別器
#====================================
#------------------------------------
# light-weight GAN の 識別器
#------------------------------------
class InputLayer(nn.Module):
    def __init__(self, in_dims, n_fmaps, image_size = 1024 ):
        super(InputLayer, self).__init__()
        if( image_size == 256 ):
            self.layers = nn.Sequential(
                spectral_norm( nn.Conv2d(in_dims, n_fmaps//4, kernel_size=3, stride=1, padding=1, bias=False) ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif( image_size == 512 ):
            self.layers = nn.Sequential(
                spectral_norm( nn.Conv2d(in_dims, n_fmaps//4, kernel_size=4, stride=2, padding=1, bias=False) ),
                nn.LeakyReLU(0.2, inplace=True),
            )
        elif( image_size == 1024 ):
            self.layers = nn.Sequential(
                spectral_norm( nn.Conv2d(in_dims, n_fmaps//8, kernel_size=4, stride=2, padding=1, bias=False) ),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm( nn.Conv2d(n_fmaps//8, n_fmaps//4, kernel_size=4, stride=2, padding=1, bias=False) ),
                nn.BatchNorm2d(n_fmaps//4),
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
            spectral_norm( nn.Conv2d(in_dims, out_dims, kernel_size=4, stride=2, padding=1, bias=False) ),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm( nn.Conv2d(out_dims, out_dims, kernel_size=3, stride=1, padding=1, bias=False) ),
            nn.BatchNorm2d(out_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.AvgPool2d((h,w)),
            spectral_norm( nn.Conv2d(in_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False) ),
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
            spectral_norm( nn.Conv2d(in_dims, out_dims, kernel_size=4, stride=1, padding=0, bias=False) ),            
            Swish(),
            spectral_norm( nn.Conv2d(out_dims, out_dims, kernel_size=1, stride=1, padding=0, bias=False) ),                        
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

class UpsamplingLayer(nn.Module):
    def __init__(self, in_dims, out_dims, scale_factor = 2, mode = "nearest"):
        super(UpsamplingLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample( scale_factor = scale_factor, mode = mode ),
            spectral_norm( nn.Conv2d(in_dims, out_dims*2, kernel_size=3, stride=1, padding=1, bias=False) ),
            nn.BatchNorm2d(out_dims*2),
            GLU(split_dim=1),
        )
        return

    def forward(self, input):
        output = self.layers(input)
        return output

class SimpleDecoder(nn.Module):
    def __init__(self, in_dims, out_dims = 3, crop_h = 8, crop_w = 8 ):
        super(SimpleDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((crop_h, crop_w)), # crop 処理有無で同等の処理になるようにするための GAP
            UpsamplingLayer(in_dims, in_dims//2),
            UpsamplingLayer(in_dims//2, in_dims//4),
            UpsamplingLayer(in_dims//4, in_dims//8),
            UpsamplingLayer(in_dims//8, in_dims//16),
            spectral_norm( nn.Conv2d(in_dims//16, out_dims, kernel_size=3, stride=1, padding=1, bias=False) ),
            nn.Tanh(),
        )
        return

    def forward(self, input):
        output = self.layers(input)
        return output

class OutputLayer(nn.Module):
    def __init__(self, in_dims, out_dims = 1 ):
        super(OutputLayer, self).__init__()
        self.layers = nn.Sequential(
            spectral_norm( nn.Conv2d(in_dims, in_dims, kernel_size=1, stride=1, padding=0, bias=False) ),
            nn.BatchNorm2d(in_dims),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm( nn.Conv2d(in_dims, out_dims, kernel_size=4, stride=1, padding=0, bias=False) ),
        )
        return

    def forward(self, input):
        output = self.layers(input)
        return output

class LightweightGANDiscriminator(nn.Module):
    """
    light-weight GAN の識別器
    """
    def __init__( self, in_dim = 3, n_fmaps = 64, image_size = 1024 ):
        super(LightweightGANDiscriminator, self).__init__()
        self.crop_rand_idx = 0

        # 入力層
        self.input_layer = InputLayer(in_dim, n_fmaps, image_size )

        # conv + ave pooling の２パスでのダウンサンプリング層
        self.down_256 = DownBlock(n_fmaps//4, n_fmaps//2)
        self.down_128 = DownBlock(n_fmaps//2, n_fmaps)
        self.down_64 = DownBlock(n_fmaps, n_fmaps*2)
        self.down_32 = DownBlock(n_fmaps*2, n_fmaps*4)
        self.down_16 = DownBlock(n_fmaps*4, n_fmaps*8)

        self.down_16_res128 = nn.Sequential(
            spectral_norm( nn.Conv2d(in_dim, n_fmaps//2, kernel_size=4, stride=2, padding=1, bias=False) ),
            nn.LeakyReLU(0.2, inplace=True),
            DownBlock(n_fmaps//2, n_fmaps),
            DownBlock(n_fmaps, n_fmaps*2),
            DownBlock(n_fmaps*2, n_fmaps*4),
        )

        # SLE
        self.sle_256_32 = SLEblock(n_fmaps//4, n_fmaps*2)
        self.sle_128_16 = SLEblock(n_fmaps//2, n_fmaps*4)
        self.sle_64_8 = SLEblock(n_fmaps, n_fmaps*8)

        # decoder 層
        #self.decoder_f1 = SimpleDecoder(n_fmaps*4, 3)
        self.decoder_f1 = SimpleDecoder(n_fmaps*2, 3)
        self.decoder_f2 = SimpleDecoder(n_fmaps*8, 3)
        self.decoder_res128 = SimpleDecoder(n_fmaps*4, 3)

        # 出力層
        self.output_layer = OutputLayer(n_fmaps*8, 1)
        self.output_res128_layer = spectral_norm( nn.Conv2d(n_fmaps*4, 1, 4, 1, 0, bias=False) )
        return

    def random_crop(self, image_tsr):
        hw = image_tsr.shape[2]//2
        if self.crop_rand_idx == 0 :
            return image_tsr[:,:,:hw,:hw]
        if self.crop_rand_idx == 1 :
            return image_tsr[:,:,:hw,hw:]
        if self.crop_rand_idx == 2 :
            return image_tsr[:,:,hw:,:hw]
        if self.crop_rand_idx == 3 :
            return image_tsr[:,:,hw:,hw:]

    def forward(self, input, input_res128 = None, interpolate_mode = "bilinear" ):
        if input_res128 is None : 
            input_res128 = F.interpolate(input, size=128, mode = interpolate_mode)

        #print( "[LightweightGANDiscriminator] input.shape", input.shape )
        #-------------------------------------
        # 入力層
        #-------------------------------------
        feat256 = self.input_layer(input)
        #print( "[LightweightGANDiscriminator] feat256.shape", feat256.shape )

        #-------------------------------------
        # down sampling & SLE 層
        #-------------------------------------
        feat128 = self.down_256(feat256)
        #print( "[LightweightGANDiscriminator] feat128.shape", feat128.shape )
        feat64 = self.down_128(feat128)
        feat32 = self.down_64(feat64)
        feat32_skip = self.sle_256_32(feat256, feat32)
        #print( "[LightweightGANDiscriminator] feat32_skip.shape", feat32_skip.shape )
        feat16 = self.down_32(feat32_skip)
        feat16_skip = self.sle_128_16(feat128, feat16)
        feat8 = self.down_16(feat16_skip)
        #print( "[LightweightGANDiscriminator] feat8.shape", feat8.shape )
        feat8_skip = self.sle_64_8(feat64, feat8)
        #print( "[LightweightGANDiscriminator] feat8_skip.shape", feat8_skip.shape )

        feat8_res128 = self.down_16_res128(input_res128)
        #print( "[LightweightGANDiscriminator] feat8_res128.shape", feat8_res128.shape )

        #-------------------------------------
        # 出力層
        #-------------------------------------
        d_output_resmax = self.output_layer(feat8_skip)
        d_output_res128 = self.output_res128_layer(feat8_res128)
        #print( "[LightweightGANDiscriminator] d_output_resmax.shape", d_output_resmax.shape )
        #print( "[LightweightGANDiscriminator] d_output_res128.shape", d_output_res128.shape )
        d_output = torch.cat( [d_output_resmax, d_output_res128], dim = 1)
        #print( "[LightweightGANDiscriminator] d_output.shape", d_output.shape )

        #-------------------------------------
        # decoder 層
        #-------------------------------------
        #rec_img_f1 = self.decoder_f1(feat16_skip)
        self.crop_rand_idx = random.randint(0, 3)
        if self.crop_rand_idx == 0 :
            rec_img_f1 = self.decoder_f1(feat32_skip[:,:,:8,:8])
        if self.crop_rand_idx == 1 :
            rec_img_f1 = self.decoder_f1(feat32_skip[:,:,:8,8:])
        if self.crop_rand_idx == 2 :
            rec_img_f1 = self.decoder_f1(feat32_skip[:,:,8:,:8])
        if self.crop_rand_idx == 3 :
            rec_img_f1 = self.decoder_f1(feat32_skip[:,:,8:,8:])

        rec_img_f2 = self.decoder_f2(feat8_skip)
        rec_img_res128 = self.decoder_res128(feat8_res128)
        #print( "[LightweightGANDiscriminator] rec_img_f1.shape", rec_img_f1.shape )
        #print( "[LightweightGANDiscriminator] rec_img_f2.shape", rec_img_f2.shape )
        #print( "[LightweightGANDiscriminator] rec_img_res128.shape", rec_img_res128.shape )

        #-------------------------------------
        # return 値
        #-------------------------------------
        outputs = {
            "d_output" : d_output,
            "d_output_resmax" : d_output_resmax,
            "d_output_res128" : d_output_res128,
            "rec_img_f1" : rec_img_f1,
            "rec_img_f2" : rec_img_f2,
            "rec_img_res128" : rec_img_res128,
        }
        return outputs
