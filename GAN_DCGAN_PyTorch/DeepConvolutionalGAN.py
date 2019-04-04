# -*- coding:utf-8 -*-

"""
    更新情報
    [19/04/03] : 新規作成
    [xx/xx/xx] : 
               : 
"""

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim


class Generator( nn.Module ):
    """
    DCGAN の生成器 G [Generator] 側のネットワーク構成を記述したモデル。

    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 使用デバイス
        _fc_layer : <nn.Sequential> 生成器の全結合層 （ノイズデータの次元を拡張）
        _deconv_layer : <nn.Sequential> 生成器の DeConv 処理を行う層
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        device
    ):
        super( Generator, self ).__init__()
        self._device = device

        # 全結合層 [fully connected layer]
        # 62×1 の入力ノイズ z を、6272 = 128 * 7 * 7 の次元まで拡張
        self._fc_layer = nn.Sequential(
            nn.Linear(62, 1024),        # 入力ノイズ x = 62pixel
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
        )

        # DeConv 処理を行う層
        # 7 * 7 * 128 → 14 * 14 * 64 → 28 * 28 * 1
        self._deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
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
        x = self._fc_layer(input)
        x = x.view(-1, 128, 7, 7)
        output = self._deconv_layer(x)
        return output


class Discriminator( nn.Module ):
    """
    DCGAN の識別器 D [Generator] 側のネットワーク構成を記述したモデル。

    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 使用デバイス
        _conv_layer : <nn.Sequential> 識別器の Conv 処理を行う層
        _fc_layer : <nn.Sequential> 識別器の全結合層 
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
       self,
       device
    ):
        super( Discriminator, self ).__init__()
        self._device = device

        self._conv_layer = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        self._fc_layer = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        self.init_weight()

        return

    def init_weight( self ):
        """
        独自の重みの初期化処理
        """
        return

    def forward(self, input):
        x = self._conv_layer(input)
        x = x.view(-1, 128 * 7 * 7)
        output = self._fc_layer(x)
        return output


class DeepConvolutionalGAN( object ):
    """
    DCGAN [Deep Convolutional GAN] を表すクラス
    --------------------------------------------
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 実行デバイス

        _learnig_rate : <float> 最適化アルゴリズムの学習率
        _batch_size : <int> ミニバッチ学習時のバッチサイズ

        _generator : <nn.Module> DCGAN の生成器
        _discriminator : <nn.Module> DCGAN の識別器

        _G_optimizer : <torch.optim.Optimizer> 生成器の最適化アルゴリズム
        _D_optimizer : <torch.optim.Optimizer> 識別器の最適化アルゴリズム

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        device,
        learing_rate = 0.0001,
        batch_size = 32
    ):
        self._device = device

        self._learning_rate = learing_rate
        self._batch_size = batch_size

        self._generator = None
        self._dicriminator = None
        self._G_optimizer = None
        self._D_optimizer = None

        self.model()
        self.loss()
        self.optimizer()

        return

    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "DeepConvolutionalGAN" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_learning_rate :", self._learning_rate )
        print( "_batch_size :", self._batch_size )
        print( "_generator :", self._generator )
        print( "_dicriminator :", self._dicriminator )
        print( "_G_optimizer :", self._G_optimizer )
        print( "_D_optimizer :", self._D_optimizer )
        print( "----------------------------------" )
        return


    @property
    def device( self ):
        """ device の Getter """
        return self._device

    @device.setter
    def device( self, device ):
        """ device の Setter """
        self._device = device
        return


    def model( self ):
        """
        モデルの定義を行う。

        [Args]
        [Returns]
        """
        self._generator = Generator( self._device )
        self._dicriminator = Discriminator( self._device )
        return

    def loss( self ):
        """
        損失関数の設定を行う。
        [Args]
        [Returns]
        """
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
            lr = self._learning_rate
        )

        self._G_optimizer = optim.Adam(
            params = self._dicriminator.parameters(),
            lr = self._learning_rate
        )

        return


    def fit( self ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Args]
        [Returns]
        """
        return

