# -*- coding:utf-8 -*-
import os
import numpy as np
import functools
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

#==================================
# StyleGAN の生成器
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

#----------------------------------
# Mapping Network 関連
#----------------------------------
class PixelNormLayer(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


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


class MappingNetwork(nn.Module):
    """
    StyleGAN の MappingNetwork f
    """
    def __init__( self, n_in_channles=512, n_out_channles=512, n_layers=8 ):
        super( MappingNetwork, self ).__init__()
        self.n_layers = n_layers
        self.pixnel_norm = PixelNormLayer()
        self.fc_layers = nn.ModuleDict(
            { "fc_{}".format(i+1) : 
                nn.Sequential(
                    SLinear( n_in_channles, n_out_channles ),
                    nn.LeakyReLU(negative_slope=0.2),
                ) for i in range(n_layers)
            }
        )
        return

    def forward( self, input ):
        output = self.pixnel_norm(input)
        for i in range(self.n_layers):
            output = self.fc_layers["fc_{}".format(i+1)](output)

        return output

#----------------------------------
# Synthesis Network 関連
#----------------------------------
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

# "learned affine transform" A
class FC_A(nn.Module):
    '''
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    '''
    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        # "the biases associated with ys that we initialize to one"
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        # Gain scale factor and bias with:
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style
    
# AdaIn (AdaptiveInstanceNorm)
class AdaIn(nn.Module):
    '''
    adaptive instance normalization
    '''
    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)
        
    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias  
        return result


class ApplyNoizeMapLayer(nn.Module):
    def __init__( self, n_fmaps=512 ):
        super( ApplyNoizeMapLayer, self ).__init__()
        self.weight = nn.Parameter(torch.zeros((1, n_fmaps, 1, 1)))

    def forward( self, input, noize_map ):
        output = input + self.weight * noize_map
        return output


class SynthesisHeadBlock(nn.Module):
    def __init__( self, in_dim_noize=512, in_dim_latent=512, out_dim=512, h=4, w=4 ):
        super( SynthesisHeadBlock, self ).__init__()
        self.adain_A1_style = FC_A(in_dim_latent, in_dim_noize)
        self.adain_A2_style = FC_A(in_dim_latent, in_dim_noize)
        self.adain = AdaIn(in_dim_noize)

        self.input_const = nn.Parameter(torch.ones(1, in_dim_noize, h, w))
        self.apply_noize_map_B1 = quick_scale( ApplyNoizeMapLayer(in_dim_noize) )
        self.apply_noize_map_B2 = quick_scale( ApplyNoizeMapLayer(in_dim_noize) )

        self.activate = nn.LeakyReLU(0.2)
        self.conv = SConv2d(in_dim_noize, out_dim, 3, padding=1)
        return

    def forward( self, latent_w, noize_map ):
        # ノイズマップ（B）からの入力
        input_const = self.input_const.expand(noize_map.shape[0], -1, -1, -1)
        noize_B1 = self.apply_noize_map_B1( input_const, noize_map )
        noize_B2 = self.apply_noize_map_B2( input_const, noize_map )
        #print( "[SynthesisHeadBlock] input_const.shape", input_const.shape )
        #print( "[SynthesisHeadBlock] noize_B1.shape", noize_B1.shape )
        #print( "[SynthesisHeadBlock] noize_B2.shape", noize_B2.shape )

        # AdaIN A1
        adain_A1_style = self.adain_A1_style(latent_w)
        adain_A1 = self.adain(noize_B1, adain_A1_style)
        #print( "[SynthesisHeadBlock] adain_A1.shape", adain_A1.shape )

        # conv
        conv = self.activate(adain_A1)
        conv = self.conv(conv)
        #print( "[SynthesisHeadBlock] conv.shape", conv.shape )

        # AdaIN A2
        adain_A2_style = self.adain_A2_style(latent_w)
        adain_A2 = self.adain(conv, adain_A2_style)
        #print( "[SynthesisHeadBlock] adain_A2.shape", adain_A2.shape )
        output = self.activate(adain_A2)
        return output


class SynthesisBlock(nn.Module):
    def __init__( self, in_dim_noize=512, in_dim_latent=512, out_dim=512 ):
        super( SynthesisBlock, self ).__init__()
        self.adain_A1_style = FC_A(in_dim_latent, out_dim)
        self.adain_A2_style = FC_A(in_dim_latent, out_dim)
        self.adain = AdaIn(out_dim)

        self.apply_noize_map_B1 = quick_scale( ApplyNoizeMapLayer(out_dim) )
        self.apply_noize_map_B2 = quick_scale( ApplyNoizeMapLayer(out_dim) )

        self.activate = nn.LeakyReLU(0.2)
        self.conv1 = SConv2d(in_dim_noize, out_dim, 3, padding=1)
        self.conv2 = SConv2d(out_dim, out_dim, 3, padding=1)
        return

    def forward( self, latent_w, noize_map, input ):
        # アップサンプリング
        output = F.interpolate(input, scale_factor=2, mode='bilinear', align_corners=False)
        #print( "[SynthesisBlock] output.shape", output.shape )
        output = self.conv1(output)

        # ノイズマップ（B）からの入力
        noize_B1 = self.apply_noize_map_B1( output, noize_map )
        noize_B2 = self.apply_noize_map_B2( output, noize_map )
        #print( "[SynthesisBlock] output.shape", output.shape )
        #print( "[SynthesisBlock] noize_B1.shape", noize_B1.shape )
        #print( "[SynthesisBlock] noize_B2.shape", noize_B2.shape )

        # AdaIN A1
        adain_A1_style = self.adain_A1_style(latent_w)
        adain_A1 = self.adain(noize_B1, adain_A1_style)
        #print( "[SynthesisBlock] adain_A1.shape", adain_A1.shape )

        # conv
        output = self.activate(adain_A1)
        output = self.conv2(output)
        #print( "[SynthesisBlock] output.shape", output.shape )

        # AdaIN A2
        adain_A2_style = self.adain_A2_style(latent_w)
        adain_A2 = self.adain(output, adain_A2_style)
        #print( "[SynthesisBlock] adain_A2.shape", adain_A2.shape )
        output = self.activate(adain_A2)
        #print( "[SynthesisBlock] output.shape", output.shape )
        return output


class SynthesisNetwork(nn.Module):
    def __init__( self, in_dim_latent=512, out_dim=3, image_size_init = 4, image_size_final = 1024 ):
        super( SynthesisNetwork, self ).__init__()
        self.progress_init = int(np.log2(image_size_init)) - 2
        self.progress_final = int(np.log2(image_size_final)) -2

        in_dims = [in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent//2, in_dim_latent//4, in_dim_latent//8, in_dim_latent//16]
        out_dims = [in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent//2, in_dim_latent//4, in_dim_latent//8, in_dim_latent//16, in_dim_latent//32]
        print( "[SynthesisNetwork] in_dims : ", in_dims )
        print( "[SynthesisNetwork] out_dims : ", out_dims )

        self.synthesis_head = SynthesisHeadBlock(in_dim_noize=in_dim_latent, in_dim_latent=in_dim_latent, out_dim=in_dim_latent )
        self.synthesis_bodys = nn.ModuleDict(
            { "synthesis_{}".format(i+1) : 
                SynthesisBlock(in_dim_noize=in_dim_fmap, in_dim_latent=in_dim_latent, out_dim=out_dim_fmap) for i, (in_dim_fmap, out_dim_fmap) in enumerate(zip(in_dims,out_dims))
            }
        )
        self.out_layers = nn.ModuleDict(
            { "conv_{}".format(i+1) : 
                SConv2d(in_dim_fmap, out_dim, 1) for i, in_dim_fmap in enumerate(in_dims)
            }
        )
        return

    def forward( self, latent_z, noize_map_list, progress = 0, alpha = 0.0 ):
        if( progress == 0 ):
            output = self.synthesis_head(latent_z, noize_map_list[0])
            output = self.out_layers["conv_{}".format(progress+1)](output)
        else:
            output_list = []

            # １段目
            output = self.synthesis_head(latent_z, noize_map_list[0])
            output_list.append(output)

            # ２段目以降
            for i in range(progress):
                #print( "[SynthesisNetwork] progress={}, i={}, synthesis_{}".format(progress, i, i+1) )
                if( i+1 >= progress ):
                    output = self.synthesis_bodys["synthesis_{}".format(i+1)](latent_z, noize_map_list[i+1], output_list[-1])
                    output_list.append(output)
                    break
                else:
                    output = self.synthesis_bodys["synthesis_{}".format(i+1)](latent_z, noize_map_list[i+1], output_list[-1])
                    output_list.append(output)
                    continue

            output = self.out_layers["conv_{}".format(progress+1)](output_list[-1])
            if( 0.0 < alpha < 1.0 ):
                output_prev = F.interpolate(output_list[-2], scale_factor=2, mode='bilinear', align_corners=False)
                output_prev = self.out_layers["conv_{}".format(progress)](output_prev)
                output = alpha * output + (1 - alpha) * output_prev

            #print( "[SynthesisNetwork] output.shape : ", output.shape )

        return output


class StyleGANGenerator(nn.Module):
    def __init__( self, in_dim=512, out_dim=3, n_mappling_layers=8 ):
        super( StyleGANGenerator, self ).__init__()
        self.mapping_network = MappingNetwork(in_dim, in_dim, n_mappling_layers)
        self.synthesis_network = SynthesisNetwork()
        return

    def truncation_trick( self, latent_w, latent_w_center, psi = 0.0 ):
        latent_w = latent_w_center + psi * (latent_w - latent_w_center) 
        return latent_w

    def forward( self, latent_z, noize_map_list, progress = 0, alpha = 0.0, truncation_trick = False ):
        latent_w = self.mapping_network(latent_z)
        #print( "[StyleGANGenerator] latent_w.shape : ", latent_w.shape )
        if( truncation_trick ):
            latent_w = self.truncation_trick( latent_w, latent_w, psi = 0.5)    # dummy

        output = self.synthesis_network(latent_w, noize_map_list, progress, alpha)
        #print( "[StyleGANGenerator] output.shape : ", output.shape )
        return output
