# -*- coding:utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
from math import ceil

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
import torchvision.utils as vutils
import tensorboardX as tbx

#
from BaseNetwork import *


class Generator( nn.Module ):
    """
    PGGAN の生成器 G [Generator] 側のネットワーク構成を記述したモデル。

    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 使用デバイス
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
        self,
        device,
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
        super( Generator, self ).__init__()
        self._device = device

        #
        self.pre = PixelNormLayer().to( self._device )

        #=======================================
        # 特徴ベクトルからRGBへの変換ネットワーク
        #=======================================
        # 4 × 4
        self.toRGBs = nn.ModuleList().to( self._device )
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        # 8 × 8
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        # 16 × 16
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        # 32 × 32
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_rgb, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer(pre_layer = layers[-1]).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.toRGBs.append( layers )

        #print( "toRGBs :\n", toRGBs )

        #=======================================
        # 0.0 < α <= 1.0 での deconv 層
        #=======================================
        self.blocks = nn.ModuleList().to( self._device )

        #---------------------------------------
        # 4 × 4 の解像度の画像生成用ネットワーク
        #---------------------------------------
        layers = []
        # conv 4 × 4 : shape = [n_fmaps, 1, 1] →　[n_fmaps, 4, 4]
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=4, stride=1, padding=3 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )

        # conv 3 × 3 : shape = [n_fmaps, 4, 4] →　[n_fmaps, 4, 4]
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #---------------------------------------
        # 8 × 8 の解像度の画像生成用ネットワーク
        #---------------------------------------
        layers = []
        # conv 3 × 3 : shape = [n_fmaps, 8, 8] →　[n_fmaps, 8, 8]
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )

        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #---------------------------------------
        # 16 × 16 の解像度の画像生成用ネットワーク
        #---------------------------------------
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        # 32 × 32 の解像度の画像生成用ネットワーク
        layers = []
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )
        layers.append( nn.Conv2d( in_channels=n_input_noize_z, out_channels=n_input_noize_z, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( PixelNormLayer().to( self._device ) )
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
            output = self.pre(input)
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


class Discriminator( nn.Module ):
    """
    PGGAN の識別器 D [Discriminator] 側のネットワーク構成を記述したモデル。

    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 使用デバイス
        _layer : <nn.Sequential> 識別器のネットワーク構成
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
        self,
        device,
        init_image_size = 4,
        final_image_size = 32,
        n_fmaps = 128,
        n_rgb = 3,
    ):
        super( Discriminator, self ).__init__()
        self._device = device

        #==============================================
        # RGB から 特徴マップ数への変換を行うネットワーク
        #==============================================
        self.fromRGBs = nn.ModuleList()

        # 4 × 4
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.fromRGBs.append( layers )

        # 8 × 8
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.fromRGBs.append( layers )

        # 16 × 16
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.fromRGBs.append( layers )

        # 32 × 32
        layers = []
        layers.append( nn.Conv2d( in_channels=n_rgb, out_channels=n_fmaps, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
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
        layers.append( nn.Conv2d( in_channels=n_fmaps+1, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )

        # conv 4 × 4 : shape = [n_fmaps, 4, 4] → [n_fmaps, 1, 1]
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=4, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )

        # conv 1 × 1 : shape = [n_fmaps, 1, 1] → [1, 1, 1]
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=1, kernel_size=1, stride=1, padding=0 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.Sigmoid().to( self._device ) )

        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #-----------------------------------------
        # 8 × 8
        #-----------------------------------------
        layers = []

        # conv 3 × 3 : [n_fmaps, 8, 8] → []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )

        # conv 3 × 3 : [n_fmaps, 8, 8] → []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #-----------------------------------------
        # 16 × 16
        #-----------------------------------------
        layers = []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers = nn.Sequential( *layers )
        self.blocks.append( layers )

        #-----------------------------------------
        # 32 × 32
        #-----------------------------------------
        layers = []
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
        layers.append( nn.Conv2d( in_channels=n_fmaps, out_channels=n_fmaps, kernel_size=3, stride=1, padding=1 ).to( self._device ) )
        layers.append( WScaleLayer( pre_layer = layers[-1] ).to( self._device ) )
        layers.append( nn.LeakyReLU(negative_slope=0.2).to( self._device ) )
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


class ProgressiveGANforMNIST( object ):
    """
    PGGAN を表すクラス
    --------------------------------------------
    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 実行デバイス
        _n_epoches : <int> エポック数（学習回数）
        _learnig_rate : <float> 最適化アルゴリズムの学習率
        _batch_size : <int> ミニバッチ学習時のバッチサイズ
        _n_input_noize_z : <int> 入力ノイズ z の次元数
        _init_image_size : <int> 最初の Training Progresses での生成画像の解像度
        final_image_size : <int> 最終的な Training Progresses での生成画像の解像度
        _generator : <nn.Module> 生成器
        _discriminator : <nn.Module> 識別器
        _loss_fn : 損失関数
        _G_optimizer : <torch.optim.Optimizer> 生成器の最適化アルゴリズム
        _D_optimizer : <torch.optim.Optimizer> 識別器の最適化アルゴリズム
        _loss_G_historys : <list> 生成器 G の損失関数値の履歴（イテレーション毎）
        _loss_D_historys : <list> 識別器 D の損失関数値の履歴（イテレーション毎）
        _images_historys : <list> 生成画像のリスト
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
        self,
        device,
        n_epoches = 10,
        learing_rate = 0.0001,
        batch_size = 32,
        n_input_noize_z = 128,
        init_image_size = 4,
        final_image_size = 32,
        n_samples = 64
    ):
        self._device = device

        self._n_epoches = n_epoches
        self._learning_rate = learing_rate
        self._batch_size = batch_size
        self._n_input_noize_z = n_input_noize_z
        self._init_image_size = init_image_size
        self._final_image_size = final_image_size

        self._generator = None
        self._dicriminator = None
        self._loss_fn = None
        self._G_optimizer = None
        self._D_optimizer = None
        self._loss_G_historys = []
        self._loss_D_historys = []
        self._images_historys = []
        self.model()
        self.loss()
        self.optimizer()
        self._fixed_input_noize_z = torch.rand( (n_samples, self._n_input_noize_z, 1, 1) ).to( self._device )
        return

    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "ProgressiveGANforMNIST" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_n_epoches :", self._n_epoches )
        print( "_learning_rate :", self._learning_rate )
        print( "_batch_size :", self._batch_size )
        print( "_n_input_noize_z :", self._n_input_noize_z )
        print( "_init_image_size :", self._init_image_size )
        print( "_final_image_size :", self._final_image_size )
        print( "_generator :", self._generator )
        print( "_dicriminator :", self._dicriminator )
        print( "_loss_fn :", self._loss_fn )
        print( "_G_optimizer :", self._G_optimizer )
        print( "_D_optimizer :", self._D_optimizer )
        print( "----------------------------------" )
        return

    @property
    def loss_G_history( self ):
        return self._loss_G_historys

    @property
    def loss_D_history( self ):
        return self._loss_D_historys

    def model( self ):
        """
        モデルの定義を行う。
        [Args]
        [Returns]
        """
        self._generator = Generator( 
            device = self._device,
            init_image_size = self._init_image_size,
            final_image_size = self._final_image_size,
            n_input_noize_z = self._n_input_noize_z,
            n_rgb = 1,
        )

        self._dicriminator = Discriminator( 
            self._device,
            init_image_size = self._init_image_size,
            final_image_size = self._final_image_size,
            n_fmaps = self._n_input_noize_z,
            n_rgb = 1,
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
        # GeneratorとDiscriminatorはそれぞれ別のOptimizerがある
        # PyTorchはOptimizerの更新対象となるパラメータを第1引数で指定することになっている（TensorFlowやKerasにはなかった）
        # この機能のおかげで D_optimizer.step() でパラメータ更新を走らせたときに、
        # Discriminatorのパラメータしか更新されず、Generatorのパラメータは固定される。
        # これにより TensorFlow における、
        # tf.control_dependencies(...) : sess.run で実行する際のトレーニングステップの依存関係（順序）を定義
        # に対応する処理が簡単にわかりやすく行える。
        self._G_optimizer = optim.Adam(
            params = self._generator.parameters(),
            lr = self._learning_rate,
            betas = (0.5,0.999)
        )

        self._D_optimizer = optim.Adam(
            params = self._dicriminator.parameters(),
            lr = self._learning_rate,
            betas = (0.5,0.999)
        )

        return

    def fit( self, dloader, n_sava_step = 5, result_path = "./result" ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Args]
            n_sava_step : <int> 学習途中での生成画像の保存間隔（イテレーション単位）
            result_path : <str> 学習途中＆結果を保存するディレクトリ
        [Returns]
        """
        if( os.path.exists( result_path ) == False ):
            os.mkdir( result_path )

        # tensor board の writer / tensorboard --logdir runs/
        writer = tbx.SummaryWriter()

        # 教師信号（０⇒偽物、1⇒本物）
        # real ラベルを 1 としてそして fake ラベルを 0 として定義
        ones_tsr =  torch.ones( self._batch_size ).to( self._device )
        zeros_tsr =  torch.zeros( self._batch_size ).to( self._device )

        #-------------------------------------
        # モデルを学習モードに切り替える。
        #-------------------------------------
        self._generator.train()
        self._dicriminator.train()

        #-------------------------------------
        # 学習処理ループ
        #-------------------------------------
        iterations = 0      # 学習処理のイテレーション回数

        init_progress = float(np.log2(self._init_image_size)) - 2
        final_progress = float(np.log2(self._final_image_size)) -2

        print("Starting Training Loop...")

        # エポック数分トレーニング
        for epoch in tqdm( range(self._n_epoches), desc = "Epoches" ):
            # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
            for i, (images,targets) in enumerate( tqdm( dloader, desc = "minbatch process in DataLoader" ) ):
                iterations += 1

                x = (epoch + i / len(dloader))
                progress = min(max(int(x / 2), x - ceil(x / 2), 0), final_progress)

                # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する
                # （後の計算で、shape の不一致をおこすため）
                if images.size()[0] != self._batch_size:
                    break

                # ミニバッチデータを GPU へ転送
                images = images.to( self._device )

                # 元の画像ピクセル数を、Training Progresses 用にダウンサンプリング
                # progress = 0.0 ⇒ 32 × 32 → 4 × 4, 
                # 0.00 < progress < 1.00 ⇒ 32 × 32 → 8 × 8,
                # 1.00 < progress < 2.00 ⇒ 32 × 32 → 16 × 16,
                # ...
                images = F.adaptive_avg_pool2d(images, 4 * 2 ** int(ceil(progress)) )

                #====================================================
                # 識別器 D の fitting 処理
                #====================================================
                # 生成器 G に入力するノイズ z
                input_noize_z = torch.rand( (self._batch_size, self._n_input_noize_z, 1,1) ).to( self._device )

                #----------------------------------------------------
                # 勾配を 0 に初期化
                # （この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
                #----------------------------------------------------
                self._D_optimizer.zero_grad()

                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
                #----------------------------------------------------
                # D(x) : 本物画像 x = image を入力したときの識別器の出力 (0.0 ~ 1.0)
                D_x = self._dicriminator( images, progress=progress )
                #print( "D_x.size() :", D_x.size() )
                #print( "D_x :", D_x )

                # G(z) : 生成器から出力される偽物画像
                G_z = self._generator( input_noize_z, progress=progress )     # torch.Size([32, 3, 4, 4])
                #print( "G_z.size() :", G_z.size() )
                #print( "G_z :", G_z )

                # D( G(z) ) : 偽物画像を入力したときの識別器の出力 (0.0 ~ 1.0)
                # 識別器 D のみ学習を行っている段階ので、生成器からの出力 G_z を deatch() して、生成器側に勾配が伝わらないようにする。
                D_G_z = self._dicriminator( G_z.detach(), progress=progress )
                #print( "D_G_z.size() :", D_G_z.size() )
                #print( "D_G_z :", D_G_z )

                #----------------------------------------------------
                # 損失関数を計算する
                # 出力と教師データを損失関数に設定し、誤差 loss を計算
                # この設定は、損失関数を __call__ をオーバライト
                # loss は Pytorch の Variable として帰ってくるので、これをloss.data[0]で数値として見る必要があり
                #----------------------------------------------------
                # E[ log{D(x)} ]
                loss_D_real = self._loss_fn( D_x, ones_tsr )
                #print( "loss_D_real : ", loss_D_real.item() )

                # E[ 1 - log{D(G(z))} ]
                loss_D_fake = self._loss_fn( D_G_z, zeros_tsr )
                #print( "loss_D_fake : ", loss_D_fake.item() )

                # 識別器 D の損失関数 = E[ log{D(x)} ] + E[ 1 - log{D(G(z))} ]
                loss_D = loss_D_real + loss_D_fake
                #print( "loss_D : ", loss_D.item() )

                self._loss_D_historys.append( loss_D.item() )
                writer.add_scalar('loss/loss_D', loss_D.item(), iterations )
                writer.add_scalar('loss/loss_D_real', loss_D_real.item(), iterations )
                writer.add_scalar('loss/loss_D_fake', loss_D_fake.item(), iterations )

                #----------------------------------------------------
                # 誤差逆伝搬
                #----------------------------------------------------
                loss_D.backward()

                #----------------------------------------------------
                # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
                #----------------------------------------------------
                self._D_optimizer.step()

                #====================================================
                # 生成器 G の fitting 処理
                #====================================================
                # 生成器 G に入力するノイズ z
                #input_noize_z = torch.rand( (self._batch_size, self._n_input_noize_z, 1, 1) ).to( self._device )

                #----------------------------------------------------
                # 勾配を 0 に初期化
                # （この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
                #----------------------------------------------------
                self._G_optimizer.zero_grad()

                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
                #----------------------------------------------------
                # G(z) : 生成器から出力される偽物画像
                #G_z = self._generator( input_noize_z )

                # D( G(z) ) : 偽物画像を入力したときの識別器の出力 (0.0 ~ 1.0)
                D_G_z = self._dicriminator( G_z, progress=progress )
                #print( "D_G_z.size() :", D_G_z.size() )

                #----------------------------------------------------
                # 損失関数を計算する
                #----------------------------------------------------
                # L_G = E[ log{D(G(z))} ]
                loss_G = self._loss_fn( D_G_z, ones_tsr )
                #print( "loss_G :", loss_G )
                self._loss_G_historys.append( loss_G.item() )
                writer.add_scalar('loss/loss_G', loss_G.item(), iterations )

                #----------------------------------------------------
                # 誤差逆伝搬
                #----------------------------------------------------
                loss_G.backward()

                #----------------------------------------------------
                # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
                #----------------------------------------------------
                self._G_optimizer.step()

                #----------------------------------------------------
                # 学習過程での自動生成画像
                #----------------------------------------------------
                # 特定のイテレーションでGeneratorから画像を保存
                if( iterations % n_sava_step == 0 ):
                    images = self.generate_fixed_images( progress = progress, b_transformed = False )
                    save_image( tensor = images, filename = result_path + "/PGGANforMNIST_Image_epochs{}_iters{}.png".format( epoch, iterations ) )
                    writer.add_image( 'Image', vutils.make_grid(images, normalize=True, scale_each=True), iterations )

            images = self.generate_fixed_images( progress = progress, b_transformed = False )
            save_image( tensor = images, filename = result_path + "/PGGANforMNIST_Image_epochs{}_iters{}.png".format( epoch, iterations ) )
            writer.add_image( 'Image', vutils.make_grid(images, normalize=True, scale_each=True), iterations )
            
        print("Finished Training Loop.")
        #writer.export_scalars_to_json( "runs/tensorboard_all_scalars.json" )
        writer.close()
        return


    def generate_fixed_images( self, progress, b_transformed = False ):
        """
        GAN の Generator から、固定された画像データを自動生成する。
        [Args]
            b_transformed : <bool> 画像のフォーマットを Tensor から変換するか否か
            progress : <float> 現在の training progress
        [Returns]
            images : <Tensor> / shape = [n_samples, n_channels, height, width]
                生成された画像データのリスト
                行成分は生成する画像の数 n_samples
        """
        # 生成器を推論モードに切り替える。
        self._generator.eval()

        # 画像を生成
        images = self._generator( self._fixed_input_noize_z, progress = progress )
        #print( "images.size() :", images.size() )

        if( b_transformed == True ):
            # Tensor → numpy に変換
            images = images.cpu().detach().numpy()

        return images
