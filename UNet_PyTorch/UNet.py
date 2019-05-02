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
from torchvision.utils import save_image


class UNet( nn.Module ):
    """
    UNet を表すクラス

    [Args]
        n_in_channels : <int> 入力画像のチャンネル数
        n_out_channels : <int> 出力画像のチャンネル数
        n_channels : <int> 特徴マップの枚数
    """
    def __init__(
        self,
        device,
        n_in_channels = 3,
        n_out_channels = 3,
        n_fmaps = 64
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
        self._conv1 = conv_block( n_in_channels, n_fmaps ).to( self._device )
        self._pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )
        self._conv2 = conv_block( n_fmaps*1, n_fmaps*2 ).to( self._device )
        self._pool2 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )
        self._conv3 = conv_block( n_fmaps*2, n_fmaps*4 ).to( self._device )
        self._pool3 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )
        self._conv4 = conv_block( n_fmaps*4, n_fmaps*8 ).to( self._device )
        self._pool4 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )

        #
        self._bridge=conv_block( n_fmaps*8, n_fmaps*16 ).to( self._device )

        # Decoder（アップサンプリング）
        self._dconv1 = dconv_block( n_fmaps*16, n_fmaps*8 ).to( self._device )
        self._up1 = conv_block( n_fmaps*16, n_fmaps*8 ).to( self._device )
        self._dconv2 = dconv_block( n_fmaps*8, n_fmaps*4 ).to( self._device )
        self._up2 = conv_block( n_fmaps*8, n_fmaps*4 ).to( self._device )
        self._dconv3 = dconv_block( n_fmaps*4, n_fmaps*2 ).to( self._device )
        self._up3 = conv_block( n_fmaps*4, n_fmaps*2 ).to( self._device )
        self._dconv4 = dconv_block( n_fmaps*2, n_fmaps*1 ).to( self._device )
        self._up4 = conv_block( n_fmaps*2, n_fmaps*1 ).to( self._device )

        # 出力層
        self._out_layer = nn.Sequential(
		    nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),
		    nn.Tanh(),
		).to( self._device )

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
        dconv1 = self._dconv1(bridge)
        concat1 = torch.cat( [dconv1,conv4], dim=1 )
        up1 = self._up1(concat1)

        dconv2 = self._dconv2(up1)
        concat2 = torch.cat( [dconv2,conv3], dim=1 )

        up2 = self._up2(concat2)
        dconv3 = self._dconv3(up2)
        concat3 = torch.cat( [dconv3,conv2], dim=1 )

        up3 = self._up3(concat3)
        dconv4 = self._dconv4(up3)
        concat4 = torch.cat( [dconv4,conv1], dim=1 )

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
        _n_channels : <int> 入力画像と出力画像ののチャンネル数
        _n_fmaps : <int> 特徴マップの枚数

        _model : <nn.Module> UNet のネットワーク構成
        _loss_fn : 損失関数
        _optimizer : <torch.optim.Optimizer> 最適化アルゴリズム
        _loss_historys : <list> 損失関数値の履歴（イテレーション毎）

        _fixed_pre_image : <Tensor> image-to-image の変換前の固定された画像（学習後の画像生成の入力画像として利用）
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__( 
        self,
        device,
        n_epoches = 50,
        learing_rate = 0.0001,
        batch_size = 64,
        n_channels = 3,
        n_fmaps = 64
    ):
        self._device = device
        self._n_epoches = n_epoches
        self._learning_rate = learing_rate
        self._batch_size = batch_size
        self._n_channels = n_channels
        self._n_fmaps = n_fmaps
        
        self._model = None
        self._loss_fn = None
        self._optimizer = None
        self._loss_historys = []
        self._images_historys = []

        self.model()
        self.loss()
        self.optimizer()

        self._fixed_pre_image = None
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
        print( "_n_channels :", self._n_channels )
        print( "_n_fmaps :", self._n_fmaps )

        print( "_model :", self._model )
        print( "_loss_fn :", self._loss_fn )
        print( "_optimizer :", self._optimizer )
        print( "----------------------------------" )
        return

    @property
    def loss_history( self ):
        return self._loss_historys

    @property
    def images_historys( self ):
        return self._images_historys

    def model( self ):
        """
        モデルの定義を行う。
        [Args]
        [Returns]
        """
        self._model = UNet(
            device = self._device,
            n_in_channels = self._n_channels,
            n_out_channels = self._n_channels,
            n_fmaps = self._n_fmaps
        )
        return

    def loss( self ):
        """
        損失関数の設定を行う。
        [Args]
        [Returns]
        """
        self._loss_fn = nn.MSELoss()
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

    def fit( self, dloader, n_sava_step = 100 ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Args]
            dloader : <DataLoader> 学習用データセットの DataLoader
            n_sava_step : <int> 学習途中での生成画像の保存間隔（エポック単位）
        [Returns]
        """
        # image-to-image の変換前の固定された画像（学習後の画像生成の入力画像として利用）の初期化済みフラグ
        b_init_fixed_image = False

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
                # shape = torch.Size([64, 3, 256, 512])
                #images = images.to( self._device )

                # 学習用データには、左側に衛星画像、右側に地図画像が入っているので、chunk で切り分ける
                # torch.chunk() : 渡したTensorを指定した個数に切り分ける。
                # pre_image : image-to-image 変換タスクでの変換前の画像
                # after_image : image-to-image 変換タスクでの変換後の画像
                pre_image, after_image = torch.chunk( images, chunks=2, dim=3 )
                pre_image = pre_image.to( self._device )
                after_image = after_image.to( self._device )
                #satel_image.requires_grad_()
                #map_image.requires_grad_()

                #----------------------------------------------------
                # 勾配を 0 に初期化
                # （この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
                #----------------------------------------------------
                self._optimizer.zero_grad()

                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
                #----------------------------------------------------
                output = self._model( pre_image )
                #print( "output.size() :", output.size() )   # torch.Size([1, 3, 256, 256])

                #----------------------------------------------------
                # 損失関数を計算する
                # 出力と教師データを損失関数に設定し、誤差 loss を計算
                # この設定は、損失関数を __call__ をオーバライト
                # loss は Pytorch の Variable として帰ってくるので、これをloss.data[0]で数値として見る必要があり
                #----------------------------------------------------
                loss = self._loss_fn( output, after_image )
                print( "loss :", loss.item() )
                self._loss_historys.append( loss.item() )

                #----------------------------------------------------
                # 誤差逆伝搬
                #----------------------------------------------------
                loss.backward()

                #----------------------------------------------------
                # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
                #----------------------------------------------------
                self._optimizer.step()

                #----------------------------------------------------
                # 学習過程での自動生成画像
                #----------------------------------------------------
                # 特定のイテレーションでGeneratorから画像を保存
                if( iterations % n_sava_step == 0 ):
                    """
                    if( b_init_fixed_image == False ):
                        b_init_fixed_image = True
                        # 初回ループの画像を、image-to-image の変換前の固定された画像（学習後の画像生成の入力画像として利用）として採用
                        self._fixed_pre_image = pre_image

                    image = self.generate_fixed_images( pre_image = self._fixed_pre_image, b_transformed = False )
                    save_image( tensor = image, filename = "UNet_Image_epoches{}_iters{}.png".format( epoch, iterations ) )
                    # [batc_size, n_channels, height, width] → [height, width, n_channels]
                    image_reshaped = image[0, :, :, :].squeeze().transpose( 0 ,2 )
                    self._images_historys.append( image_reshaped.cpu().detach().numpy() )
                    """
                    save_image( tensor = output.cpu(), filename = "UNet_Image_epoches{}_iters{}.png".format( epoch, iterations ) )

            #----------------------------------------------------
            # 学習過程での自動生成画像
            #----------------------------------------------------
            n_sava_epoch_step = 1
            # 特定のエポックでGeneratorから画像を保存
            if( epoch % n_sava_epoch_step == 0 ):
                """
                if( b_init_fixed_image == False ):
                    b_init_fixed_image = True
                    # 初回ループの画像を、image-to-image の変換前の固定された画像（学習後の画像生成の入力画像として利用）として採用
                    self._fixed_pre_image = pre_image

                image = self.generate_fixed_images( pre_image = self._fixed_pre_image, b_transformed = False )
                save_image( tensor = image, filename = "UNet_Image_epoches{}_iters{}.png".format( epoch, iterations ) )
                # [batc_size, n_channels, height, width] → [height, width, n_channels]
                image_reshaped = image[0, :, :, :].squeeze().transpose( 0 ,2 )
                self._images_historys.append( image_reshaped.cpu().detach().numpy() )
                """
                save_image( tensor = output.cpu(), filename = "UNet_Image_epoches{}_iters{}.png".format( epoch, iterations ) )

        #self.save_image_historys_gif( file_name = "UNet_Image_epoches{}_iters{}.gif".format( epoch, iterations ) )
        print("Finished Training Loop.")
        return


    def generate_fixed_images( self, pre_image, b_transformed = False ):
        """
        U-Net から、固定された画像データを自動生成する。
        [Args]
            pre_image : <Tensor> image-to-image の変換前の画像
            b_transformed : <bool> 画像のフォーマットを Tensor から変換するか否か
        [Returns]
            images : <Tensor> / shape = [n_channels, height, width]
                生成された画像データのリスト
        """
        # GPU に転送
        pre_image = pre_image.to( self._device )

        # 生成器を推論モードに切り替える。
        self._model.eval()

        # 画像を生成
        images = self._model( pre_image )

        if( b_transformed == True ):
            # Tensor → numpy に変換
            images = images.cpu().detach().numpy()

        return images


    def save_image_historys_gif( self, file_name = "UNet.gif" ):
        """
        学習中の生成画像の様子を gif ファイルで保存
        """
        import imageio
        imageio.mimsave( file_name, self._images_historys )
        return