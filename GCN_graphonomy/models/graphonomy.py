# -*- coding:utf-8 -*-
import os
import numpy as np
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torchvision

from models.backbone import ResNet, Bottleneck
from models.aspp import ASPP
from models.decoder import Decoder
from models.gcn import GraphConvolution, FeatureMaptoGraphProjection, GraphtoFeatureMapProjection, InterGraphTransfer

#====================================
# Graphonomy 関連のネットワーク
#====================================
class GraphonomyIntraGraphReasoning( nn.Module ):
    """
    Graphonomy の １つのデータセットでの Intra-Graph Reasoning
    """
    def __init__( self, n_in_channels = 3, n_classes = 20, n_node_features = 128 ):
        super(GraphonomyIntraGraphReasoning, self).__init__()
        self.backbone = ResNet( block = Bottleneck, layers = [3, 4, 23, 3], n_in_channels = n_in_channels, output_stride = 16, BatchNorm = nn.BatchNorm2d, pretrained = True )
        self.aspp = ASPP( backbone = "resnet", output_stride = 16, BatchNorm = nn.BatchNorm2d, dropout = 0.0 )
        self.decoder = Decoder( num_classes = n_classes, backbone = "resnet", BatchNorm = nn.BatchNorm2d, dropout1 = 0.0, dropout2 = 0.0, use_semantic_conv = False )
        self.feature_to_graph_proj = FeatureMaptoGraphProjection( in_features = 256, out_features = n_node_features, n_nodes = n_classes )
        self.graph_conv1 = GraphConvolution( in_features = n_node_features, out_features = n_node_features, activate = True, sparse = False )
        self.graph_conv2 = GraphConvolution( in_features = n_node_features, out_features = n_node_features, activate = True, sparse = False )
        self.graph_conv3 = GraphConvolution( in_features = n_node_features, out_features = n_node_features, activate = True, sparse = False )
        self.graph_to_feature_proj = GraphtoFeatureMapProjection( in_features = 256, out_features = 256, n_hiddens = n_node_features, n_nodes = n_classes )
        self.skip_conv = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(True)])
        self.semantic_conv = nn.Conv2d(256, n_classes, kernel_size=1)
        return

    def forward(self, input, adj_matrix = None ):
        #print( "input.shape : ", input.shape )
        #------------------------------
        # DeepLab v3+ での特徴量抽出処理
        #------------------------------
        # backbone ネットワーク（ResNet）で projection
        backbone, backbone_low_level = self.backbone(input)
        #print( "backbone.shape : ", backbone.shape )                        # torch.Size([2, 2048, 32, 32])
        #print( "backbone_low_level.shape : ", backbone_low_level.shape )    # torch.Size([2, 256, 128, 128])

        # ASPP で各畳み込みを統合
        embedded = self.aspp(backbone)
        #print( "[encoder] embedded.shape : ", embedded.shape )              # torch.Size([2, 256, 32, 32])

        # decode 処理
        embedded = self.decoder(embedded, backbone_low_level)
        #print( "[decoder] embedded.shape : ", embedded.shape )              # torch.Size([2, 256, 128, 128])

        #---------------------------------------------
        # Intra-Graph Reasoning での処理
        #---------------------------------------------
        # DeepLab v3+ で enocder-decoder した特徴マップをグラフ構造に射影する
        graph = self.feature_to_graph_proj(embedded)
        #print( "graph.shape : ", graph.shape )                              # torch.Size([2, 20, 128])

        # グラフ構造をグラフ畳み込み
        graph = self.graph_conv1(graph, adj_matrix)
        graph = self.graph_conv2(graph, adj_matrix)
        graph = self.graph_conv3(graph, adj_matrix)
        #print( "graph.shape : ", graph.shape )                              # torch.Size([adj_shape[0], 2, 20, 128])

        # グラフ構造を特徴マップに射影する
        reproj_feature = self.graph_to_feature_proj(graph,embedded)        
        #print( "reproj_feature.shape : ", reproj_feature.shape )            # torch.Size([2, 256, 128, 128])

        # skip connection
        skip = self.skip_conv(embedded)
        skip = skip + reproj_feature
        #print( "skip.shape : ", skip.shape )                                 # torch.Size([2, 256, 128, 128])

        # セマンティクス形式の出力
        semantic = self.semantic_conv(skip)
        #print( "semantic.shape : ", semantic.shape )                        # torch.Size([2, 20, 128, 128])

        # upsampling
        semantic_upsample = F.interpolate(semantic, size=input.size()[2:], mode='bilinear', align_corners=True)
        #print( "semantic_upsample.shape : ", semantic_upsample.shape )       # torch.Size([2, 20, 512, 512])

        return semantic_upsample, embedded, graph, reproj_feature


class Graphonomy( nn.Module ):
    """
    Graphonomy の ２つのデータセットでの Intra-Graph Reasoning & Inter-Graph Transfer
    """
    def __init__( self, n_in_channels = 3, n_classes_source = 7, n_classes_target = 20, n_node_features = 128 ):
        super(Graphonomy, self).__init__()
        self.backbone = ResNet( block = Bottleneck, layers = [3, 4, 23, 3], n_in_channels = n_in_channels, output_stride = 16, BatchNorm = nn.BatchNorm2d, pretrained = True )
        self.aspp = ASPP( backbone = "resnet", output_stride = 16, BatchNorm = nn.BatchNorm2d, dropout = 0.0 )
        self.decoder = Decoder( num_classes = n_classes_source, backbone = "resnet", BatchNorm = nn.BatchNorm2d, dropout1 = 0.0, dropout2 = 0.0, use_semantic_conv = False )
        self.feature_to_graph_proj = FeatureMaptoGraphProjection( in_features = 256, out_features = n_node_features, n_nodes = n_classes_source )
        self.source_graph_conv1 = GraphConvolution( in_features = n_node_features, out_features = n_node_features, activate = True, sparse = False )
        self.source_graph_conv2 = GraphConvolution( in_features = n_node_features, out_features = n_node_features, activate = True, sparse = False )
        self.source_graph_conv3 = GraphConvolution( in_features = n_node_features, out_features = n_node_features, activate = True, sparse = False )
        self.source_to_target_graph_trans = InterGraphTransfer( in_features = n_node_features, out_features = n_node_features, n_nodes_source = n_classes_source, n_nodes_target = n_classes_target, bias = False, adj_matrix = None, activate = True )
        self.source_graph_to_feature_proj = GraphtoFeatureMapProjection( in_features = 256, out_features = 256, n_hiddens = n_node_features, n_nodes = n_classes_source )
        self.source_skip_conv = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=1), nn.ReLU(True)])
        self.source_semantic_conv = nn.Conv2d(256, n_classes_source, kernel_size=1)
        return

    def forward(self, input, adj_matrix_source = None, adj_matrix_target = None, adj_matrix_transfer = None ):
        """
        [Args]
            input : <Tensor> 入力画像
            adj_matrix_source : <Tensor> 変換元グラフ構造の隣接行列
            adj_matrix_target : <Tensor> 変換先グラフ構造の隣接行列
            adj_matrix_transfer : <Tensor> 別のグラフ構造への隣接行列
        """
        #print( "input.shape : ", input.shape )
        #------------------------------
        # DeepLab v3+ での特徴量抽出処理
        #------------------------------
        # backbone ネットワーク（ResNet）で encoder
        backbone, backbone_low_level = self.backbone(input)
        #print( "backbone.shape : ", backbone.shape )                        # torch.Size([2, 2048, 32, 32])
        #print( "backbone_low_level.shape : ", backbone_low_level.shape )    # torch.Size([2, 256, 128, 128])

        # ASPP で各畳み込みを統合
        embedded = self.aspp(backbone)
        #print( "embedded.shape : ", embedded.shape )                        # torch.Size([2, 256, 32, 32])

        # decode 処理
        embedded = self.decoder(embedded, backbone_low_level)
        #print( "embedded.shape : ", embedded.shape )                        # torch.Size([2, 256, 128, 128])

        #--------------------------------------------
        # １つ目のデータセットでの Inter-Graph Reasoning での処理
        #--------------------------------------------
        # DeepLab v3+ で enocder-decoder した特徴マップをグラフ構造に射影する
        graph = self.feature_to_graph_proj(embedded)
        #print( "graph.shape : ", graph.shape )                              # torch.Size([2, 7, 128])

        # 変換元グラフ構造をグラフ畳み込み
        source_graph1 = self.source_graph_conv1(graph, adj_matrix_source)
        source_graph2 = self.source_graph_conv2(graph, adj_matrix_source)
        source_graph3 = self.source_graph_conv3(graph, adj_matrix_source)
        #print( "source_graph1.shape : ", source_graph1.shape )              # torch.Size([adj_shape[0], 2, 7, 128])
        #print( "source_graph2.shape : ", source_graph2.shape )
        #print( "source_graph3.shape : ", source_graph3.shape )
        
        # グラフ構造を特徴マップに射影する
        source_reproj_feature = self.source_graph_to_feature_proj(source_graph3, embedded)        
        #print( "source_reproj_feature.shape : ", source_reproj_feature.shape )            # torch.Size([2, 256, 128, 128])

        # skip connection
        source_skip = self.source_skip_conv(embedded)
        source_skip = source_skip + source_reproj_feature
        #print( "source_skip.shape : ", source_skip.shape )                                 # torch.Size([2, 256, 128, 128])

        # セマンティクス形式の出力
        source_semantic = self.source_semantic_conv(source_skip)
        #print( "source_semantic.shape : ", source_semantic.shape )                        # torch.Size([2, 7, 128, 128])

        # upsampling
        source_semantic = F.interpolate(source_semantic, size=input.size()[2:], mode='bilinear', align_corners=True)
        #print( "source_semantic.shape : ", source_semantic.shape )       # torch.Size([2, 20, 512, 512])

        #--------------------------------------------
        # ２つ目のデータセットでの Inter-Graph Transfer での処理
        #--------------------------------------------
        # 変換元グラフ構造から変換先グラフ構造への変換
        source_to_target_graph1 = self.source_to_target_graph_trans( source_graph1, adj_matrix = adj_matrix_transfer )
        source_to_target_graph2 = self.source_to_target_graph_trans( source_graph2, adj_matrix = adj_matrix_transfer )
        source_to_target_graph3 = self.source_to_target_graph_trans( source_graph3, adj_matrix = adj_matrix_transfer )
        #print( "source_to_target_graph1.shape : ", source_to_target_graph1.shape )  # torch.Size([1, 2, 20, 128])
        #print( "source_to_target_graph2.shape : ", source_to_target_graph2.shape )  # torch.Size([1, 2, 20, 128])
        #print( "source_to_target_graph3.shape : ", source_to_target_graph3.shape )  # torch.Size([1, 2, 20, 128])

        #--------------------------------------------
        # ２つ目のデータセットでの Inter-Graph Reasoning での処理
        #--------------------------------------------
        # 変換先グラフ構造をグラフ畳み込み
        pass

        #--------------------------------------------
        # Inter-Graph Transfer での処理
        #--------------------------------------------
        #　グラフの結合
        pass

        return source_semantic, embedded, source_graph3
