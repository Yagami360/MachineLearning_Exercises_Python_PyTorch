# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
import torchvision.models as models

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#=======================================
# Geometric-matching CNN
#=======================================
class FeatureExtraction( nn.Module ):
    """
    Geometric-matching CNN の feature extraction
    画像から特徴量を抽出する
    """
    def __init__( self, pre_trained_model = 'vgg', l2_norm = True, freeze = False ):
        super(FeatureExtraction, self).__init__()
        self.l2_norm = l2_norm

        # 事前学習済みモデル
        if( pre_trained_model == 'vgg' ):
            self.model = models.vgg16(pretrained=True)
            # VGG の pool4 までのネットワークを利用
            #print( "[VGG]", self.model.features.children() )
            self.model = nn.Sequential( *list(self.model.features.children())[:24] )
        elif( pre_trained_model == 'resnet101' ):
            self.model = models.resnet101(pretrained=True)
            # resnet101 の layer3 までのネットワークを利用
            #print( "[ResNet101]", self.model.children() )
            self.model = nn.Sequential(
                self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool,
                self.model.layer1, self.model.layer2, self.model.layer3,              
            )
        elif( pre_trained_model == 'resnet101_v2' ):
            self.model = models.resnet101(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        elif( pre_trained_model == 'densenet201' ):
            self.model = models.densenet201(pretrained=True)
            # keep feature extraction network up to transitionlayer2
            self.model = nn.Sequential(*list(self.model.features.children())[:-4])
        else:
            NotImplementedError()

        if( freeze ):
            # モデルのパラメータの更新を不可にする
            for param in self.model.parameters():
                param.requires_grad = False

        return

    def featureL2Norm( self, feature, epsilon = 1e-6 ):
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

    def forward( self, input ):
        features = self.model(input)
        if( self.l2_norm) :
            features = self.featureL2Norm(features)

        return features


class FeatureCorrelation(torch.nn.Module):
    def __init__( self, matching_type = 'correlation', shape = '3D', l2_norm = True ):
        """
        [Args]

        """
        super(FeatureCorrelation, self).__init__()
        self.matching_type = matching_type
        self.shape = shape
        self.l2_norm = l2_norm
        self.relu = nn.ReLU()
        return

    def featureL2Norm( self, feature, epsilon = 1e-6 ):
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)

    def forward( self, feature_A, feature_B ):
        # 内積で類似度を計算
        if( self.matching_type == "correlation" ):
            b,c,h,w = feature_A.size()
            if( self.shape == '3D' ):
                # feature_A : [B,C,H,W] -> [B, C, W*H] / feature_B : [B,C,H,W] -> [B, H*W, C]
                feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
                feature_B = feature_B.view(b,c,h*w).transpose(1,2)
                # correlation = f_B ^T * f_A
                correlation = torch.bmm(feature_B,feature_A)
                # indexed [batch,idx_A=row_A+h*col_A,row_B,col_B]
                correlation = correlation.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
            elif( self.shape == '4D' ):
                # reshape features for matrix multiplication
                feature_A = feature_A.view(b,c,h*w).transpose(1,2) # size [b,c,h*w]
                feature_B = feature_B.view(b,c,h*w) # size [b,c,h*w]
                # perform matrix mult.
                correlation = torch.bmm(feature_A,feature_B)
                # indexed [batch,row_A,col_A,row_B,col_B]
                correlation = correlation.view(b,h,w,h,w).unsqueeze(1)

            if( self.l2_norm ):
                # relu で負の値を０にして L2 Norm
                correlation = self.featureL2Norm( self.relu(correlation) )

        # 差分で類似度を計算
        elif( self.matching_type == "subtraction" ):
            correlation = feature_A.sub(feature_B)

        # concat で類似度を計算
        elif( self.matching_type == "concatenation" ):
            correlation = torch.cat( (feature_A,feature_B),1 )
        else:
            NotImplementedError()

        return correlation

class FeatureRegression(nn.Module):
    def __init__( self, n_channels = [225,128,64], kernel_sizes = [7,5,5], n_out_channels = 6, batch_norm = True ):
        super(FeatureRegression, self).__init__()
        n_layers = len(kernel_sizes)
        modules = []
        for i in range(n_layers-1): # last layer is linear 
            modules.append( nn.Conv2d(n_channels[i], n_channels[i+1], kernel_size = kernel_sizes[i], padding=0) )
            if batch_norm:
                modules.append( nn.BatchNorm2d(n_channels[i+1]) )
            modules.append( nn.ReLU(inplace=True) )

        self.model = nn.Sequential(*modules)        
        self.fc_layer = nn.Linear(n_channels[-1] * kernel_sizes[-1] * kernel_sizes[-1], n_out_channels )
        return

    def forward(self, input):
        output = self.model(input)

        # [B,C,H,W] -> [B,C*H*W]
        output = output.view(output.shape[0], -1)

        # 全結合層でパラメータ回帰
        output = self.fc_layer(output)
        return output


class GeometricMatchingCNN( nn.Module ):
    """
    Geometric-matching CNN / 論文「[Geometric-matching CNN] Convolutional neural network architecture for geometric matching」
    """
    def __init__( self, n_out_channels = 6, pre_trained_model = 'vgg', matching_type = 'correlation', l2_norm = True ):
        super(GeometricMatchingCNN, self).__init__()
        self.feature_extraction_A = FeatureExtraction( pre_trained_model = pre_trained_model, l2_norm = l2_norm, freeze = False )
        self.feature_extraction_B = FeatureExtraction( pre_trained_model = pre_trained_model, l2_norm = l2_norm, freeze = False )
        self.feature_correlation = FeatureCorrelation( matching_type = matching_type, shape = '3D', l2_norm = l2_norm )
        self.feature_regression = FeatureRegression( n_out_channels = n_out_channels, batch_norm = True )
        return

    def forward( self, input_A, input_B ):
        # ２つの画像から特徴量を抽出
        feature_A = self.feature_extraction_A(input_A)
        feature_B = self.feature_extraction_B(input_B)

        # 抽出した２つの特徴量の類似度を計算
        correlation = self.feature_correlation(feature_A, feature_B)

        # ２つの特徴量の類似を幾何変換パラメータ θ に回帰
        theta = self.feature_regression(correlation)
        return theta
