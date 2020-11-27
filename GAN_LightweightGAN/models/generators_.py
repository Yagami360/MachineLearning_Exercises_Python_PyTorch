# -*- coding:utf-8 -*-
import os
import numpy as np
import functools

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import torchvision

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)        

class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)

class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(4), 
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)

class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        spectral_norm(nn.ConvTranspose2d(nz, channel*2, 4, 1, 0, bias=False)),
                        batchNorm2d(channel*2),
                        GLU()
                        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)

def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block

def UpBlockComp(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        batchNorm2d(out_planes*2), GLU()
        )
    return block

def ConvBlock(in_planes, out_planes):
    block = nn.Sequential(
        conv2d(in_planes, out_planes, 3, 1, 1, bias=False),
        batchNorm2d(out_planes),
        nn.LeakyReLU(0.2))
    return block


class LightweightGANGenerator(nn.Module):
    """docstring for CAN_Generator"""

    def __init__(self, ngf=64, nz=100, nc=3, im_size=1024):
        super(LightweightGANGenerator, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=ngf*16)
                                
        self.feat_8 = UpBlockComp(nfc[4], nfc[8])
        self.feat_16 = UpBlock(nfc[8], nfc[16])
        self.feat_32 = UpBlockComp(nfc[16], nfc[32])
        self.feat_64 = UpBlock(nfc[32], nfc[64])
        self.feat_128 = UpBlockComp(nfc[64], nfc[128])  
        self.feat_256 = UpBlock(nfc[128], nfc[256]) 

        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = nn.Sequential( conv2d(nfc[128], nc, 1, 1, 0, bias=False) , nn.Tanh() )
        self.to_big = nn.Sequential( conv2d(nfc[im_size], nc, 3, 1, 1, bias=False) , nn.Tanh() )
        
        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512]) 
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])  
        
    def forward(self, input):
        
        feat_4 = self.init(input)
        feat_8 = self.feat_8(feat_4)
        feat_16 = self.feat_16(feat_8)
        feat_32 = self.feat_32(feat_16)
        feat_64 = self.feat_64(feat_32)
        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )
        feat_256 = self.se_256( feat_16, self.feat_256(feat_128) )
        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]
        
        feat_512 = self.se_512( feat_32, self.feat_512(feat_256) )
        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)
        return [self.to_big(feat_1024), self.to_128(feat_128)]

class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBlock(out_planes, out_planes)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        if im_size == 1024:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, ndf//8, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(ndf//8, ndf//4, 4, 2, 1, bias=False),
                                    batchNorm2d(ndf//4),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, ndf//4, 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential( 
                                    conv2d(nc, ndf//4, 3, 1, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )

        self.down_4  = DownBlock(ndf//4, ndf//2)
        self.down_8  = DownBlock(ndf//2, ndf*1)
        self.down_16 = DownBlock(ndf*1,  ndf*2)
        self.down_32 = DownBlock(ndf*2,  ndf*4)
        self.down_64 = DownBlock(ndf*4,  ndf*16)

        self.rf_big = nn.Sequential(
                            conv2d(ndf*16 , ndf*16, 1, 1, 0, bias=False),
                            batchNorm2d(ndf*16),
                            nn.LeakyReLU(0.2, inplace=True),
                            conv2d(ndf * 16, 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(ndf//4, ndf*2)
        self.se_4_32 = SEBlock(ndf//2, ndf*4)
        self.se_8_64 = SEBlock(ndf*1, ndf*16)
        
        self.down_from_small = nn.Sequential( conv2d(nc, ndf//2, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(ndf//2,  ndf*1),
                                            DownBlock(ndf*1,  ndf*2),
                                            DownBlock(ndf*2,  ndf*4), )
        self.rf_from_128 = conv2d(ndf*4, 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(ndf*16, nc)
        self.decoder_part = SimpleDecoder(ndf*4, nc)
        self.decoder_small = SimpleDecoder(ndf*4, nc)
        
    def forward(self, imgs, label):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]
        feat_2 = self.down_from_big(imgs[0])        
        feat_4 = self.down_4(feat_2)
        feat_8 = self.down_8(feat_4)
        
        feat_16 = self.down_16(feat_8)
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)
        feat_32 = self.se_4_32(feat_4, feat_32)
        
        feat_last = self.down_64(feat_32)
        feat_last = self.se_8_64(feat_8, feat_last)

        rf_0 = self.rf_big(feat_last)

        feat_small = self.down_from_small(imgs[1])
        rf_1 = self.rf_from_128(feat_small)

        if label=='real':    
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            part = random.randint(0, 3)
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,8:])

            return torch.cat([rf_0, rf_1], dim=1) , [rec_img_big, rec_img_small, rec_img_part], part 

        return torch.cat([rf_0, rf_1], dim=1) 



class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)
                 
        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    UpBlock(nfc_in, nfc[16]) ,
                                    UpBlock(nfc[16], nfc[32]),
                                    UpBlock(nfc[32], nfc[64]),
                                    UpBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)