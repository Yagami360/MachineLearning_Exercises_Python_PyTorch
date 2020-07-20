# -*- coding:utf-8 -*-
import os
import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torchvision

#====================================
# Graph Convolutional Networks
#====================================
class GraphConvolution( nn.Module ):
    """
    グラフ畳み込み / 
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # nn.Parameter() でネットワークのパラメータを一括に設定
        # この nn.Parameter() で作成したデータは、普通の Tensor 型とは異なり, <class 'torch.nn.parameter.Parameter'> という別の型になる
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        return

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # torch.mm() : 行列の積 / input * self.weight
        support = torch.mm(input, self.weight)

        # torch.spmm() : 疎行列の演算 / adj : 隣接行列で疎行列になっている
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionNetworks( nn.Module ):
    """
    ３層のグラフ畳み込みネットワーク
    """
    def __init__( self, n_inputs, n_outputs, n_hiddens1 = 128, n_hiddens2 = 64, dropout = 0.25 ):
        super(GraphConvolutionNetworks, self).__init__()

        self.gc1 = GraphConvolution(n_inputs, n_hiddens1)
        self.activate1 = nn.ReLU()
        self.dropout1 = nn.Dropout( dropout )

        self.gc2 = GraphConvolution(n_hiddens1, n_hiddens2)
        self.activate2 = nn.ReLU()
        self.dropout2 = nn.Dropout( dropout )

        self.gc3 = GraphConvolution(n_hiddens2, n_outputs)
        self.activate3 = nn.LogSoftmax()
        #self.activate3 = nn.Softmax()
        return

    def forward(self, x, adj):
        out = self.gc1(x, adj)
        out = self.activate1(out)
        out = self.dropout1(out)

        out = self.gc2(out, adj)
        out = self.activate2(out)
        out = self.dropout2(out)

        out = self.gc3(out, adj)
        out = self.activate3(out)
        return out

