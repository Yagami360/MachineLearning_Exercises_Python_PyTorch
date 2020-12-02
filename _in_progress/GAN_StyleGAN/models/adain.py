# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


class AdaIN(nn.Module):
    def __init__( self, n_hin_channles, n_in_channles, net_type = "conv", norm_type = "batch", resize_type = "nearest" ):
        """
        [args]
            resize_type : ネットワーク外部から入力するデータ（セグメンテーション画像など）のサイズを、前段のネットワークからの活性化入力のサイズにリサイズして合わせるときのリサイズ手法
                'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'
        """
        super(AdaIN, self).__init__()
        self.net_type = net_type
        self.norm_type = norm_type
        self.resize_type = resize_type
        if( self.norm_type == "batch" ):
            self.norm_layer = nn.BatchNorm2d(n_hin_channles, affine=False)
        elif( self.norm_type == "instance" ):
            self.norm_layer = nn.InstanceNorm2d(n_hin_channles, affine=False)
        else:
            NotImplementedError()

        if( self.net_type == "fc" ):
            self.gamma_layer = nn.Linear(n_in_channles, n_hin_channles)
            self.beta_layer = nn.Linear(n_in_channles, n_hin_channles)
        elif( self.net_type == "conv" ):
            self.gamma_layer = nn.Conv2d(n_in_channles, n_hin_channles, kernel_size=3, stride=1, padding=1)
            self.beta_layer = nn.Conv2d(n_in_channles, n_hin_channles, kernel_size=3, stride=1, padding=1)
        else:
            NotImplementedError()

        return

    def forward(self, h_in, input ):
        """
        [args]
            h_in : 前段のネットワークからの活性化入力のテンソル
            input : ネットワーク外部から入力するデータ（セグメンテーション画像など）のテンソル
        """
        # ネットワーク外部から入力するデータ（セグメンテーション画像など）のサイズを、前段のネットワークからの活性化入力のサイズにリサイズして合わせる
        if( len(input.shape) == 4 and h_in.shape[2:] != input.shape[2:] ):
            input = F.interpolate(input, size=h_in.size()[2:], mode=self.resize_type)

        h_act = self.norm_layer(h_in)
        if( self.net_type == "fc" ):
            gamma = self.gamma_layer(input)
            beta = self.beta_layer(input)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)
        elif( self.net_type == "conv" ):
            gamma = self.gamma_layer(input)
            beta = self.beta_layer(input)
        else:
            NotImplementedError()

        #print( "[AdaIN] h_in.shape", h_in.shape )
        #print( "[AdaIN] h_act.shape", h_act.shape )
        #print( "[AdaIN] gamma.shape", gamma.shape )
        #print( "[AdaIN] beta.shape", beta.shape )
        h_after = gamma * h_act + beta
        #print( "h_after.shape", h_after.shape )
        return h_after


class AdaINResBlock(nn.Module):
    def __init__( self, n_hin_channles, n_hout_channles, n_in_channles, net_type = "conv", norm_type = "batch", resize_type = "nearest" ):
        super(AdaINResBlock, self).__init__()
        self.n_hin_channles = n_hin_channles
        self.n_hout_channles = n_hout_channles

        self.adin1 = AdaIN(n_hin_channles, n_in_channles, net_type, norm_type, resize_type )
        self.activate1 = nn.ReLU()
        self.conv1 = nn.Conv2d(n_hin_channles, n_hin_channles, kernel_size=3, stride=1, padding=1)

        self.adin2 = AdaIN(n_hin_channles, n_in_channles, net_type, norm_type, resize_type )
        self.activate2 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_hin_channles, n_hout_channles, kernel_size=3, stride=1, padding=1)

        # skip connection でチャンネル数を一致させるための畳込み
        if( self.n_hin_channles != self.n_hout_channles ):
            self.adin3 = AdaIN(n_hin_channles, n_in_channles, net_type, norm_type, resize_type )
            self.activate3 = nn.ReLU()
            self.conv3 = nn.Conv2d(n_hin_channles, n_hout_channles, kernel_size=1, bias=False)

        return

    def forward(self, h_in, input):
        """
        [args]
            h_in : 前段のネットワークからの活性化入力のテンソル
            input : ネットワーク外部から入力するデータ（セグメンテーション画像など）のテンソル
        """
        #print( "[AdaINResBlock] h_in.shape : ", h_in.shape )
        #print( "[AdaINResBlock] input.shape : ", input.shape )

        output1 = self.adin1(h_in, input)
        output1 = self.activate1(output1)
        output1 = self.conv1(output1)

        output2 = self.adin2(output1, input)
        output2 = self.activate2(output2)
        output2 = self.conv2(output2)

        if( self.n_hin_channles != self.n_hout_channles ):
            skip_connection = self.adin3(h_in, input)
            skip_connection = self.activate3(skip_connection)
            skip_connection = self.conv3(skip_connection)
        else:
            skip_connection = input

        #print( "[AdaINResBlock] output2.shape : ", output2.shape )
        #print( "[AdaINResBlock] skip_connection.shape : ", skip_connection.shape )
        output = output2 + skip_connection
        return output

