# -*- coding:utf-8 -*-

"""
    更新情報
    [19/04/30] : 新規作成
    [xx/xx/xx] : 
               : 
"""
import os
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class UNet( nn.Module ):
    """
    UNet を表すクラス

    [Args]
        in_dim : <int>
        out_dim : <int>
        n_channels : <int> 特徴マップの枚数
    """
    def __init__(
        self,
        device,
        in_dim = 3,
        out_dim = 3,
        n_channels = 64
    ):
        super( UNet, self ).__init__()
        self._device = device

        def conv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True ),

                nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
            )
            return model

        def dconv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.ConvTranspose2d( in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1 ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        # Encoder（ダウンサンプリング）
        self._conv1 = conv_block( in_dim, out_dim )
        self._pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self._conv2 = conv_block( n_channels*1, n_channels*2 )
        self._pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self._conv2 = conv_block( n_channels*2, n_channels*4 )
        self._pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self._conv2 = conv_block( n_channels*4, n_channels*8 )
        self._pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )

        #
        self._bridge=conv_block( n_channels*8, n_channels*16 )

        # Decoder（アップサンプリング）
        self._dconv1 = dconv_block( n_channels*16, n_channels*8 )
        self._up1 = conv_block( n_channels*16, n_channels*8 )
        self._dconv2 = dconv_block( n_channels*8, n_channels*4 )
        self._up2 = conv_block( n_channels*8, n_channels*4 )
        self._dconv3 = dconv_block( n_channels*4,n_channels*2 )
        self._up3 = conv_block( n_channels*4, n_channels*2 )
        self._dconv4 = dconv_block( n_channels*2, n_channels*1 )
        self._up4 = conv_block( n_channels*2, n_channels*1 )

        # 出力層
        self._out_layer = nn.Sequential(
		    nn.Conv2d( n_channels, out_dim, 3, 1, 1 ),
		    nn.Tanh(),
		)

        return

    @staticmethod
    def weights_init( model ):
        return

    def forward( self, input ):
        # Encoder（ダウンサンプリング）
        conv1 = self._conv1( input )
        pool1 = self._pool1( conv1 )
        conv2 = self._conv2( pool1 )
        pool2 = self._pool2( conv2 )
        conv3 = self._conv3( pool2 )
        pool3 = self._pool3( conv3 )
        conv4 = self._conv4( pool3 )
        pool4 = self._pool4( conv4 )

        #
        bridge = self._bridge( pool4 )

        # Decoder（アップサンプリング）& skip connection
        trans1 = self._dconv1(bridge)
        concat1 = torch.cat( [trans1,down4], dim=1 )
        up1 = self._up1(concat1)

        trans2 = self._dconv2(up1)
        concat2 = torch.cat( [trans2,down3], dim=1 )

        up2 = self._up2(concat2)
        trans3 = self._dconv3(up2)
        concat3 = torch.cat( [trans3,down2], dim=1 )

        up3 = self._up3(concat3)
        trans4 = self._dconv4(up3)
        concat4 = torch.cat( [trans4,down1], dim=1 )

        up4 = self._up4(concat4)

        # 出力層
        output = self._out_layer( up4 )

        return output


class SemanticSegmentationwithUNet( object ):
    """
    セマンティックセグメンテーションを表すクラス
    ・ネットワーク構成は、UNet
    --------------------------------------------
    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 実行デバイス
        _n_epoches : <int> エポック数（学習回数）
        _learnig_rate : <float> 最適化アルゴリズムの学習率
        _batch_size : <int> ミニバッチ学習時のバッチサイズ
        _n_channels : <int> 入力画像のチャンネル数
        _n_fmaps : <int> 特徴マップの枚数

        _model : <nn.Module> UNet のネットワーク構成
        _loss_fn : 損失関数
        _optimizer : <torch.optim.Optimizer> 最適化アルゴリズム
        _loss_historys : <list> 損失関数値の履歴（イテレーション毎）
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__( 
        self,
        device,
        n_epoches = 50,
        learing_rate = 0.0001,
        batch_size = 64
    ):
        self._device = device
        self._n_epoches = n_epoches
        self._learning_rate = learing_rate
        self._batch_size = batch_size

        self._model = None
        self._loss_fn = None
        self._optimizer = None
        self._loss_historys = []

        self.model()
        self.loss()
        self.optimizer()
        return

    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "SemanticSegmentationwithUNet" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_n_epoches :", self._n_epoches )
        print( "_learning_rate :", self._learning_rate )
        print( "_batch_size :", self._batch_size )

        print( "_model :", self._model )
        print( "_loss_fn :", self._loss_fn )
        print( "_optimizer :", self._optimizer )
        print( "----------------------------------" )
        return

    @property
    def loss_history( self ):
        return self._loss_historys

    def model( self ):
        """
        モデルの定義を行う。
        [Args]
        [Returns]
        """
        self._model = UNet(
            device = self._device
        )
        return

    def loss( self ):
        """
        損失関数の設定を行う。
        [Args]
        [Returns]
        """
        self._loss_fn = nn.BCELoss()
        return

    def optimizer( self ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        [Args]
        [Returns]
        """
        self._optimizer = optim.Adam(
            params = self._model.parameters(),
            lr = self._learning_rate,
            betas = (0.5,0.999)
        )
        return

    def fit( self, dloader, n_sava_step = 5 ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Args]
            dloader : <DataLoader> 学習用データセットの DataLoader
            n_sava_step : <int> 学習途中での生成画像の保存間隔（エポック単位）
        [Returns]
        """
        #-------------------------------------
        # モデルを学習モードに切り替える。
        #-------------------------------------
        self._model.train()

        #-------------------------------------
        # 学習処理ループ
        #-------------------------------------
        iterations = 0      # 学習処理のイテレーション回数

        print("Starting Training Loop...")
        # エポック数分トレーニング
        for epoch in tqdm( range(self._n_epoches), desc = "Epoches" ):
            # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
            for (images,targets) in tqdm( dloader, desc = "minbatch process in DataLoader" ):
                #print( "images.size() : ", images.size() )
                #print( "targets.size() : ", targets.size() )

                # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する
                # （後の計算で、shape の不一致をおこすため）
                if images.size()[0] != self._batch_size:
                    break

                iterations += 1

                # ミニバッチデータを GPU へ転送
                images = images.to( self._device )

        print("Finished Training Loop.")
        return