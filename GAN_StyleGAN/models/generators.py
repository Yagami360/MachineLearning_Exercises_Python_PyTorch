# -*- coding:utf-8 -*-
import os
import numpy as np
import functools
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

from models.adain import AdaIN


class TempleteNetworks( nn.Module ):
    """
    ダミー用の何もしないネットワーク
    """
    def __init__( self, in_dim = 3, out_dim = 1 ):
        super( TempleteNetworks, self ).__init__()
        self.dummmy_layer = nn.Sequential(
            nn.Conv2d( in_dim, out_dim, kernel_size=1, stride=1, padding=0 ),
        )
        return

    def forward( self, input ):
        if( len(input.shape) == 2 ):
            input = input.unsqueeze(-1).unsqueeze(-1)
            input = input.expand(input.shape[0], input.shape[1], 128, 128 )
            print( "input.shape : ", input.shape )

        output = self.dummmy_layer(input)
        return output

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
        self.adain_A1 = AdaIN( n_hin_channles = in_dim_noize, n_in_channles = in_dim_latent, net_type = "fc", norm_type = "instance")
        #self.adain_A1.gamma_layer.bias.data = 1
        self.adain_A1.beta_layer.bias.data.zero_()
        self.adain_A2 = AdaIN( n_hin_channles = in_dim_noize, n_in_channles = in_dim_latent, net_type = "fc", norm_type = "instance")
        #self.adain_A2.gamma_layer.bias.data = 1
        self.adain_A2.beta_layer.bias.data.zero_()

        self.input_const = nn.Parameter(torch.ones(1, in_dim_noize, h, w))
        self.apply_noize_map_B1 = ApplyNoizeMapLayer(in_dim_noize)
        self.apply_noize_map_B2 = ApplyNoizeMapLayer(in_dim_noize)

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
        adain_A1 = self.adain_A1(noize_B1, latent_w)
        #print( "[SynthesisHeadBlock] adain_A1.shape", adain_A1.shape )

        # conv
        conv = self.activate(adain_A1)
        conv = self.conv(conv)
        #print( "[SynthesisHeadBlock] adain_A1.shape", adain_A1.shape )

        # AdaIN A2
        adain_A2 = self.adain_A1(conv, latent_w)
        #print( "[SynthesisHeadBlock] adain_A2.shape", adain_A2.shape )
        output = self.activate(adain_A2)
        return output


class SynthesisBlock(nn.Module):
    def __init__( self, in_dim_noize=512, in_dim_latent=512, out_dim=512 ):
        super( SynthesisBlock, self ).__init__()
        self.adain_A1 = AdaIN( n_hin_channles = in_dim_noize, n_in_channles = in_dim_latent, net_type = "fc", norm_type = "instance")
        #self.adain_A1.gamma_layer.bias.data = 1
        self.adain_A1.beta_layer.bias.data.zero_()
        self.adain_A2 = AdaIN( n_hin_channles = in_dim_noize, n_in_channles = in_dim_latent, net_type = "fc", norm_type = "instance")
        #self.adain_A2.gamma_layer.bias.data = 1
        self.adain_A2.beta_layer.bias.data.zero_()

        self.apply_noize_map_B1 = ApplyNoizeMapLayer(in_dim_noize)
        self.apply_noize_map_B2 = ApplyNoizeMapLayer(in_dim_noize)

        self.activate = nn.LeakyReLU(0.2)
        self.conv1 = SConv2d(in_dim_noize, out_dim, 3, padding=1)
        self.conv2 = SConv2d(out_dim, out_dim, 3, padding=1)
        return

    def forward( self, latent_z, noize_map, input ):
        output = self.conv1(input)

        # ノイズマップ（B）からの入力
        noize_B1 = self.apply_noize_map_B1( output, noize_map )
        noize_B2 = self.apply_noize_map_B2( output, noize_map )

        # AdaIN A1
        adain_A1 = self.adain_A1(noize_B1, latent_w)

        # conv
        output = self.activate(adain_A1)
        output = self.conv2(output)

        # AdaIN A2
        adain_A2 = self.adain_A1(output, latent_w)
        output = self.activate(adain_A2)
        return output


class SynthesisNetwork(nn.Module):
    def __init__( self, in_dim_latent=512, out_dim=3, resolution_max=1024 ):
        super( SynthesisNetwork, self ).__init__()
        """
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1
        self.num_layers = resolution_log2 * 2 - 2
        """
        in_dims = [in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent//2, in_dim_latent//4, in_dim_latent//8, in_dim_latent//16]
        out_dims = [in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent, in_dim_latent//2, in_dim_latent//4, in_dim_latent//8, in_dim_latent//16, in_dim_latent//32]
        print( "in_dims : ", in_dims )
        print( "out_dims : ", out_dims )

        self.synthesis_head = SynthesisHeadBlock(in_dim_noize=in_dim_latent, in_dim_latent=in_dim_latent, out_dim=in_dim_latent )
        self.synthesis_bodys = nn.ModuleDict(
            { "synthesis_{}".format(i+1) : 
                SynthesisBlock(in_dim_fmap, out_dim_fmap, in_dim_latent) for i, (in_dim_fmap, out_dim_fmap) in enumerate(zip(in_dims,out_dims))
            }
        )
        self.out_layers = nn.ModuleDict(
            { "conv_{}".format(i+1) : 
                SConv2d(out_dim_fmap, out_dim, 1) for i, out_dim_fmap in enumerate(out_dims)
            }
        )
        return

    def forward( self, latent_z, noize_map, train_progress = 0 ):
        if( train_progress == 0 ):
            output = self.synthesis_head(latent_z, noize_map)
        else:
            output = self.synthesis_bodys["synthesis_{}".format(train_progress+1)](latent_z, noize_map)

        output = self.out_layers["conv_{}".format(train_progress+1)](output)
        return output


class StyleGANGenerator(nn.Module):
    def __init__( self, in_dim=512, out_dim=3, n_mappling_layers=8 ):
        super( StyleGANGenerator, self ).__init__()
        self.mapping_network = MappingNetwork(in_dim, in_dim, n_mappling_layers)
        self.synthesis_network = SynthesisNetwork()
        return

    def forward( self, latent_z, noize_map, train_progress = 0 ):
        latent_w = self.mapping_network(latent_z)
        print( "[StyleGANGenerator] latent_w.shape : ", latent_w.shape )

        output = self.synthesis_network(latent_w, noize_map, train_progress)
        print( "[StyleGANGenerator] output.shape : ", output.shape )
        return output
