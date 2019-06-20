import numpy as np

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim

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
