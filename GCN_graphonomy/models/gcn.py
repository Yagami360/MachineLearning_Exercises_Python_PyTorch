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
    def __init__(self, in_features, out_features, bias = True, activate = False, sparse = False ):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparse = sparse

        # nn.Parameter() でネットワークのパラメータを一括に設定
        # この nn.Parameter() で作成したデータは、普通の Tensor 型とは異なり, <class 'torch.nn.parameter.Parameter'> という別の型になる
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.activate = activate
        if( self.activate ):
            self.activate_layer = nn.ReLU()
        return

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj = None):
        # torch.mm() : ２次元同士の行列の積 / input * self.weight
        #support = torch.mm(input, self.weight)
        support = torch.matmul(input, self.weight)

        if adj is not None:
            if( self.sparse ):
                # torch.spmm() : 疎行列の演算 / adj : 隣接行列で疎行列になっている
                output = torch.spmm(adj, support)
            else:
                output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if( self.activate_layer ):
            output = self.activate_layer(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


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


class FeatureMaptoGraphProjection( nn.Module ):
    """
    特徴マップをグラフ構造に射影する
    射影は学習可能な重み付きの行列積で行う
    """
    def __init__(self, in_features = 256, out_features = 128, n_nodes = 20 ):
        """
        [Args]
            in_features : <int> 入力特徴マップのチャンネル数
            out_features : <int> 学習可能な重み行列 W の次元数
            n_nodes : <int> グラフの頂点数（ノード数）
        """
        super(FeatureMaptoGraphProjection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_nodes = n_nodes

        # nn.Parameter() でネットワークのパラメータを一括に設定
        # この nn.Parameter() で作成したデータは、普通の Tensor 型とは異なり, <class 'torch.nn.parameter.Parameter'> という別の型になる
        self.feature_to_graph_matrix = Parameter(torch.FloatTensor(in_features, n_nodes))
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        return

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        batch_size, n_channels, height, width = input.shape[0], input.shape[1], input.shape[2], input.shape[3]

        # (B,C,H,W) -> (N,C,H*W) -> (N,H*W,C)
        input_reshape = input.view(batch_size, n_channels, height*width).transpose(1,2)
        #print( "input_reshape.shape : ", input_reshape.shape )

        # feature map -> feature_graph / (N,H*W,C) * (in_features, n_nodes) = (N,H*W,n_nodes)
        feature_graph = torch.matmul(input_reshape, self.feature_to_graph_matrix )
        #print( "feature_graph.shape : ", feature_graph.shape )

        # feature map -> wight_graph / (N,H*W,C) * (in_features, out_features) = (N,H*W,out_features)
        wight_graph = torch.matmul(input_reshape, self.weight )
        #print( "wight_graph.shape : ", wight_graph.shape )

        #
        feature_graph = F.softmax(feature_graph, dim=-1)
        #print( "feature_graph.shape : ", feature_graph.shape )

        # feature_graph, wight_graph -> graph / (N,H*W,n_nodes) * (N,H*W,out_features) = (N,n_nodes,out_features)
        graph = F.relu( torch.matmul(feature_graph.transpose(1,2), wight_graph) )
        return graph

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.n_nodes) + ',' + str(self.out_features) + ')'


class InterGraphTransfer( nn.Module ):
    """
    Inter-Graph Transfer での異なるグラフ間での変換＆学習処理
    """
    def __init__(self, in_features = 128, out_features = 128, n_nodes_source = 7, n_nodes_target = 20, bias = False, adj_matrix = None, activate = True ):
        super(InterGraphTransfer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_nodes_source = n_nodes_source
        self.n_nodes_target = n_nodes_target

        self.weight = Parameter(torch.FloatTensor(in_features,out_features))
        if adj_matrix is not None:
            h,w = adj_matrix.size()
            assert (h == n_nodes_source) and (w == n_nodes_target)
            self.adj_matrix = torch.autograd.Variable(adj_matrix,requires_grad=False)
        else:
            self.adj_matrix = Parameter(torch.FloatTensor(n_nodes_target, n_nodes_source))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)

        self.activate = activate
        if( self.activate ):
            self.activate_layer = nn.ReLU()

        return

    def norm_trans_adj_matrix(self,adj_matrix):  # maybe can use softmax
        adj_matrix = F.relu(adj_matrix)
        r = F.softmax(adj_matrix,dim=-1)
        return r

    def forward(self, input, adj_matrix = None ):
        """
        [Args]
            input : <Tensor> 変換元グラフ構造
            adj_matrix : <Tensor> 変換元グラフ構造から変換先グラフ構造への隣接行列
        """
        support = torch.matmul( input, self.weight )

        adj_matrix_norm = self.norm_trans_adj_matrix(adj_matrix)
        output = torch.matmul(adj_matrix_norm,support)

        if adj_matrix is None:
            adj_matrix = self.adj_matrix
        if self.bias is not None:
            output = output + self.bias
        if( self.activate_layer ):
            output = self.activate_layer(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.n_nodes_source) + "," + str(self.in_features) + ' -> '  + str(self.n_nodes_target) + ',' + str(self.out_features) + ')'
