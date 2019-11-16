# -*- coding:utf-8 -*-
import torch
import torch.nn as nn

class Generator( nn.Module ):
    """
    WGAN の生成器 G [Generator] 側のネットワーク構成を記述したモデル。

    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _layer : <nn.Sequential> 生成器のネットワーク構成
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        n_input_noize_z = 100,
        n_channels = 3,
        n_fmaps = 64
    ):
        super( Generator, self ).__init__()
        
        self._layer = nn.Sequential(
            nn.ConvTranspose2d(n_input_noize_z, n_fmaps*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_fmaps*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d( n_fmaps*8, n_fmaps*4, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d( n_fmaps*4, n_fmaps*2, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d( n_fmaps*2, n_fmaps, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d( n_fmaps, n_channels, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.Tanh()
        )

        self.init_weight()
        return

    def init_weight( self ):
        """
        独自の重みの初期化処理
        """
        return

    def forward( self, input ):
        """
        ネットワークの順方向での更新処理
        ・nn.Module クラスのメソッドをオーバライト

        [Args]
            input : <Tensor> ネットワークに入力されるデータ（ノイズデータ等）
        [Returns]
            output : <Tensor> ネットワークからのテンソルの出力
        """
        output = self._layer(input)
        return output


class Critic( nn.Module ):
    """
    WGAN のクリティック側のネットワーク構成を記述したモデル。

    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _layer : <nn.Sequential> クリティックのネットワーク構成
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
       self,
       n_channels = 3,
       n_fmaps = 64
    ):
        super( Critic, self ).__init__()        
        self._layer = nn.Sequential(
            nn.Conv2d(n_channels, n_fmaps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_fmaps, n_fmaps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_fmaps*2, n_fmaps*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_fmaps*4, n_fmaps*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_fmaps*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(n_fmaps*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

        self.init_weight()

        return

    def init_weight( self ):
        """
        独自の重みの初期化処理
        """
        return

    def forward(self, input):
        # input : torch.Size([batch_size, n_channels, width, height])
        output = self._layer( input )

        # ミニバッチ平均
        output = output.mean(0)

        return output.view(1)
