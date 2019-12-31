# -*- coding:utf-8 -*-
import os
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ReshapeLayer(nn.Module):
    """
    reshape を行う層
    """
    def __init__(self, new_shape):
        """
        [Args]
            new_shape : <list : int> reshape 後の shape
        """
        super(ReshapeLayer, self).__init__()
        self._new_shape = new_shape  # not include minibatch dimension

    def forward(self, x):
        return x.view(-1, *self._new_shape)


class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()

    def forward(self, x, y):
        return torch.cat([x, y], 1)

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


class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    PGGAN が提案している安定化手法の１つである、Pixelwise feature vector normalization in generator
    PGGAN では、生成器の各畳み込み層の後の、中間層からの出力（＝特徴ベクトル）に対して、”各ピクセル毎に”、以下のような特徴ベクトルの正規化処理を行う。
    b_{x,y} = a_{x,y} / \sqrt{ 1/N * Σ_j=0^N-1 (a_{x,y}^j)^2 + ε } 
    """
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self._eps = eps
    
    def forward(self, x):
        return x / torch.sqrt( torch.mean(x ** 2, dim=1, keepdim=True) + self._eps )

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self._eps)


def resize_activations(v, so):
    """
    Resize activation tensor 'v' of shape 'si' to match shape 'so'.
    :param v:
    :param so:
    :return:
    """
    si = list(v.size())
    so = list(so)
    assert len(si) == len(so) and si[0] == so[0]

    # Decrease feature maps.
    if si[1] > so[1]:
        v = v[:, :so[1]]

    # Shrink spatial axes.
    if len(si) == 4 and (si[2] > so[2] or si[3] > so[3]):
        assert si[2] % so[2] == 0 and si[3] % so[3] == 0
        ks = (si[2] // so[2], si[3] // so[3])
        v = F.avg_pool2d(v, kernel_size=ks, stride=ks, ceil_mode=False, padding=0, count_include_pad=False)

    # Extend spatial axes. Below is a wrong implementation
    # shape = [1, 1]
    # for i in range(2, len(si)):
    #     if si[i] < so[i]:
    #         assert so[i] % si[i] == 0
    #         shape += [so[i] // si[i]]
    #     else:
    #         shape += [1]
    # v = v.repeat(*shape)
    if si[2] < so[2]: 
        assert so[2] % si[2] == 0 and so[2] / si[2] == so[3] / si[3]  # currently only support this case
        v = F.upsample(v, scale_factor=so[2]//si[2], mode='nearest')

    # Increase feature maps.
    if si[1] < so[1]:
        z = torch.zeros((v.shape[0], so[1] - si[1]) + so[2:])
        v = torch.cat([v, z], 1)
    return v

class GSelectLayer(nn.Module):
    def __init__(self, pre, chain, post):
        super(GSelectLayer, self).__init__()
        assert len(chain) == len(post)
        self.pre = pre
        self.chain = chain
        self.post = post
        self.N = len(self.chain)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.N  # cur_level: physical index
        if y is not None:
            assert insert_y_at is not None

        min_level, max_level = int(np.floor(cur_level-1)), int(np.ceil(cur_level-1))
        min_level_weight, max_level_weight = int(cur_level+1)-cur_level, cur_level-int(cur_level)
        
        _from, _to, _step = 0, max_level+1, 1

        if self.pre is not None:
            x = self.pre(x)

        out = {}
        #print('G: level=%s, size=%s' % ('in', x.size()))
        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                #print( self.chain[level] )
                x = self.chain[level](x)

            #print('G: level=%d, size=%s' % (level, x.size()))

            if level == min_level:
                out['min_level'] = self.post[level](x)
            if level == max_level:
                out['max_level'] = self.post[level](x)
                x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                        out['max_level'] * max_level_weight
        #print('G:', x.size())
        return x


class DSelectLayer(nn.Module):
    def __init__(self, pre, chain, inputs):
        super(DSelectLayer, self).__init__()
        assert len(chain) == len(inputs)
        self.pre = pre
        self.chain = chain
        self.inputs = inputs
        self.N = len(self.chain)

    def forward(self, x, y=None, cur_level=None, insert_y_at=None):
        if cur_level is None:
            cur_level = self.N  # cur_level: physical index
        if y is not None:
            assert insert_y_at is not None

        max_level, min_level = int(np.floor(self.N-cur_level)), int(np.ceil(self.N-cur_level))
        min_level_weight, max_level_weight = int(cur_level+1)-cur_level, cur_level-int(cur_level)
        
        _from, _to, _step = min_level+1, self.N, 1

        if self.pre is not None:
            x = self.pre(x)

        #print('D: level=%s, size=%s, max_level=%s, min_level=%s' % ('in', x.size(), max_level, min_level))

        if max_level == min_level:  # x = torch.Size([32, 3, 4, 4])
            x = self.inputs[max_level](x)   # x = torch.Size([32, 128, 4, 4])
            #print( "self.inputs[max_level](x) :", x.size() )
            if max_level == insert_y_at:
                x = self.chain[max_level](x, y)
            else:
                x = self.chain[max_level](x)    # torch.Size([32, 1, 1, 1])
                print('D: self.chain[max_level](x)=%s, max_level=%s, min_level=%s' % (x.size(), max_level, min_level))
        else:
            out = {}
            tmp = self.inputs[max_level](x)
            if max_level == insert_y_at:
                tmp = self.chain[max_level](tmp, y)
            else:
                tmp = self.chain[max_level](tmp)
            out['max_level'] = tmp
            out['min_level'] = self.inputs[min_level](x)
            x = resize_activations(out['min_level'], out['max_level'].size()) * min_level_weight + \
                                out['max_level'] * max_level_weight
            if min_level == insert_y_at:
                x = self.chain[min_level](x, y)
            else:
                x = self.chain[min_level](x)

        for level in range(_from, _to, _step):
            if level == insert_y_at:
                x = self.chain[level](x, y)
            else:
                x = self.chain[level](x)
        
            #print('D: level=%d, size=%s' % (level, x.size()))

        return x

#====================================
# Generators
#====================================
class ProgressiveGenerator( nn.Module ):
    """
    PGGAN の生成器 G [Generator] 側のネットワーク構成を記述したモデル。
    """
    def __init__(
        self,
        init_image_size = 4,
        final_image_size = 32,
        n_input_noize_z = 128,
        n_rgb = 3,
    ):
        """
        [Args]
            n_in_channels : <int> 入力画像のチャンネル数
            n_out_channels : <int> 出力画像のチャンネル数
        """
        super( ProgressiveGenerator, self ).__init__()

        #
        self.pre = PixelNormLayer()

        #=======================================
        # 特徴ベクトルからRGBへの変換ネットワーク
        #=======================================
        # 4 × 4
        self.toRGBs = nn.ModuleList()
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        # 8 × 8
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        # 16 × 16
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        # 32 × 32
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        #print( "toRGBs :\n", toRGBs )

        #=======================================
        # 0.0 < α <= 1.0 での deconv 層
        #=======================================
        self.blocks = nn.ModuleList()

        #---------------------------------------
        # 4 × 4 の解像度の画像生成用ネットワーク
        #---------------------------------------
        layers = []
        # conv 4 × 4 : shape = [n_fmaps, 1, 1] →　[n_fmaps, 4, 4]
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=4, stride=1, padding=3 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )

        # conv 3 × 3 : shape = [n_fmaps, 4, 4] →　[n_fmaps, 4, 4]
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #---------------------------------------
        # 8 × 8 の解像度の画像生成用ネットワーク
        #---------------------------------------
        layers = []
        # conv 3 × 3 : shape = [n_fmaps, 8, 8] →　[n_fmaps, 8, 8]
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )

        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #---------------------------------------
        # 16 × 16 の解像度の画像生成用ネットワーク
        #---------------------------------------
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        # 32 × 32 の解像度の画像生成用ネットワーク
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2) )
        layers.append( PixelNormLayer() )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #print( "blocks :\n", blocks )

        return

    def forward( self, input, progress ):
        """
        [Args]
            input : <Tensor> ネットワークへの入力
            progress : <float> 現在の Training Progress / 0.0 → 0.0 ~ 1.0 → 1.0 → 1.0 ~ 2.0 → 2.0 → ...
        """
        #-----------------------------------------
        # 学習開始時点（α=0.0）
        #-----------------------------------------
        if( progress % 1 == 0 ):
            #print( "input", input )
            output = self.pre(input)
            #print( "PixelNorm", input )
            output = self.blocks[0](output)

            for i in range(1, int(ceil(progress) + 1)):
                output = F.upsample(output, scale_factor=2)
                output = self.blocks[i](output)

            # converting to RGB
            output = self.toRGBs[int(ceil(progress))](output)

        #-----------------------------------------
        # 0.0 < α <= 1.0
        #-----------------------------------------
        else:
            alpha = progress - int(progress)
            output1 = self.pre(input)
            output1 = self.blocks[0](output1)
            #output0 = output1

            for i in range(1, int(ceil(progress) + 1)):
                output1 = F.upsample(output1, scale_factor=2)
                output0 = output1
                output1 = self.blocks[i](output1)
            
            output1 = self.toRGBs[int(ceil(progress))](output1)
            output0 = self.toRGBs[int(progress)](output0)
            output = alpha * output1 + (1 - alpha) * output0     # output0 : torch.Size([16, 1, 4, 4]), output : torch.Size([16, 1, 8, 8])

        return output


#====================================
# Discriminators
#====================================
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

    def forward(self, input, progress ):
        """
        [Args]
            input : <Tensor> ネットワークへの入力
            progress : <float> 現在の Training Progress / 0.0 → 0.0 ~ 1.0 → 1.0 → 1.0 ~ 2.0 → 2.0 → ...
        """
        #-----------------------------------------
        # 学習開始時点（α=0.0）
        #-----------------------------------------
        if( progress % 1 == 0 ):
            # shape = [1, x, x] → [n_fmaps, x, x]            
            output = self.fromRGBs[int(ceil(progress))](input)

            # shape = [n_fmaps, x, x] → [n_fmaps, 4, 4]
            for i in range(int(progress), 0, -1):
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
            alpha = progress - int(progress)

            output0 = F.avg_pool2d(input, kernel_size=2, stride=2)  # Downsampling
            output0 = self.fromRGBs[int(progress)](output0)

            output1 = self.fromRGBs[int(ceil(progress))](input)
            output1 = self.blocks[int(ceil(progress))](output1)
            output1 = F.avg_pool2d(output1, kernel_size=2, stride=2)  # Downsampling

            output = alpha * output1 + (1 - alpha) * output0

            # shape = [n_fmaps, x, x] → [n_fmaps, 4, 4]
            for i in range(int(progress), 0, -1):
                output = self.blocks[i](output)
                output = F.avg_pool2d(output, kernel_size=2, stride=2)  # Downsampling

            # shape = [n_fmaps, 4, 4] → [n_fmaps+1, 4, 4]
            output = torch.cat( ( output, self.minibatchstd(output).expand_as(output[:, 0].unsqueeze(1)) ), dim=1 )   # tmp : torch.Size([16, 129, 4, 4])

            # shape = [n_fmaps, 4, 4] → [1, 1, 1]
            output = self.blocks[0]( output )
            output = output.squeeze()

        return output