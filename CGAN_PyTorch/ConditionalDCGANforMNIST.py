# -*- coding:utf-8 -*-

"""
    更新情報
    [19/04/26] : 新規作成
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
from torchvision.utils import save_image


class Generator( nn.Module ):
    """
    CGAN の生成器 G [Generator] 側のネットワーク構成を記述したモデル。
    ・MNIST のデータ構造に最適化されている。
    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 使用デバイス
        _fc_layer : <nn.Sequential> 生成器の全結合層 （ノイズデータの次元を拡張）
        _deconv_layer : <nn.Sequential> 生成器の DeConv 処理を行う層
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）

    """
    def __init__(
        self,
        device,
        n_input_noize_z = 62
    ):
        super( Generator, self ).__init__()
        self._device = device

        n_classes = 10

        # 全結合層 [fully connected layer]
        # 入力ノイズ z を、6272 = 128 * 7 * 7 の次元まで拡張
        self._fc_layer = nn.Sequential(
            nn.Linear( n_input_noize_z + n_classes, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(),
        ).to( self._device )

        # DeConv 処理を行う層
        # 7 * 7 * 128 → 14 * 14 * 64 → 28 * 28 * 1
        self._deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        ).to( self._device )

        self.init_weight()
        return

    def init_weight( self ):
        """
        独自の重みの初期化処理
        """
        return

    def forward( self, z, y ):
        """
        ネットワークの順方向での更新処理
        ・nn.Module クラスのメソッドをオーバライト

        [Args]
            z : <Tensor> 入力ノイズ z
            y : <Tensor> クラスラベル
        [Returns]
            output : <Tensor> ネットワークからのテンソルの出力
        """
        output = torch.cat( [z, y], dim = 1 )
        output = self._fc_layer(output)
        output = output.view(-1, 128, 7, 7)
        output = self._deconv_layer(output)
        return output


class Discriminator( nn.Module ):
    """
    CGAN の識別器 D [Generator] 側のネットワーク構成を記述したモデル。
    ・MNIST のデータ構造に最適化されている。

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

        n_classes = 10
        self._conv_layer = nn.Sequential(
            nn.Conv2d(                  # shape = [batch_size, n_channels, height, width]
                in_channels = 1 + n_classes,        # インプットのチャンネルの数 
                out_channels = 64,      # アウトプットのチャンネルの数
                kernel_size = 4,        # カーネルのサイズ（＝重み行列の行数と列数）     
                stride = 2,             # ストライド幅
                padding = 1             #  Zero-padding added to both sides of the input. Default: 0
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),            
        ).to( self._device )

        self._fc_layer = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        ).to( self._device )

        self.init_weight()

        return

    def init_weight( self ):
        """
        独自の重みの初期化処理
        """
        return

    def forward(self, x, y):
        output = torch.cat( [x, y], dim=1 )
        output = self._conv_layer(output)
        output = output.view(-1, 128 * 7 * 7)
        output = self._fc_layer(output)
        return output


class ConditionalDCGANforMNIST( object ):
    """
    Conditional GAN（CGAN）を表すクラス
    ・ネットワーク構成は、DCGAN ベース
    ・MNIST のデータ構造に最適化されている。
    --------------------------------------------
    [public]

    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 実行デバイス

        _n_epoches : <int> エポック数（学習回数）
        _learnig_rate : <float> 最適化アルゴリズムの学習率
        _batch_size : <int> ミニバッチ学習時のバッチサイズ
        _n_input_noize_z : <int> 入力ノイズ z の次元数

        _generator : <nn.Module> DCGAN の生成器
        _discriminator : <nn.Module> DCGAN の識別器

        _loss_fn : <> 損失関数

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
        n_epoches = 300,
        learing_rate = 0.0001,
        batch_size = 64,
        n_input_noize_z = 62,
        n_samples = 64
    ):
        self._device = device

        self._n_epoches = n_epoches
        self._learning_rate = learing_rate
        self._batch_size = batch_size
        self._n_input_noize_z = n_input_noize_z
        self._n_classes = 10

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
        
        self._fixed_input_noize_z = torch.rand( (n_samples, self._n_input_noize_z) ).to( self._device )
        return

    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "ConditionalDCGANfroMNIST" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_n_epoches :", self._n_epoches )
        print( "_learning_rate :", self._learning_rate )
        print( "_batch_size :", self._batch_size )
        print( "_n_input_noize_z :", self._n_input_noize_z )
        print( "_n_classes :", self._n_classes )
        print( "_generator :", self._generator )
        print( "_dicriminator :", self._dicriminator )
        print( "_loss_fn :", self._loss_fn )
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

    @property
    def loss_G_history( self ):
        return self._loss_G_historys

    @property
    def loss_D_history( self ):
        return self._loss_D_historys

    @property
    def images_historys( self ):
        return self._images_historys


    def model( self ):
        """
        モデルの定義を行う。

        [Args]
        [Returns]
        """
        self._generator = Generator( self._device, n_input_noize_z = self._n_input_noize_z )
        self._dicriminator = Discriminator( self._device )
        return

    def loss( self ):
        """
        損失関数の設定を行う。
        [Args]
        [Returns]
        """
        # Binary Cross Entropy
        # L(x,y) = - { y*log(x) + (1-y)*log(1-x) }
        # x,y の設定は、後の fit() 内で行う。
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


    def fit( self, dloader, n_sava_step = 5 ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Args]
            dloader : <DataLoader> 学習用データセットの DataLoader
            n_sava_step : <int> 学習途中での生成画像の保存間隔（エポック単位）
        [Returns]
        """
        # 教師信号（０⇒偽物、1⇒本物）
        # real ラベルを 1 としてそして fake ラベルを 0 として定義
        ones_tsr =  torch.ones( self._batch_size ).to( self._device )
        zeros_tsr =  torch.zeros( self._batch_size ).to( self._device )

        # one-hot encoding 用の Tensor
        # shape = [n_classes, n_clasees]
        # [1,0,0,...,0]
        # [0,1,0,...,0]
        # ...
        # [0,0,0,...,0]
        eye_tsr = torch.eye( self._n_classes ).to( self._device )

        #-------------------------------------
        # モデルを学習モードに切り替える。
        #-------------------------------------
        self._generator.train()
        self._dicriminator.train()

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

                # 識別器に入力するクラスラベルの画像 y（＝本物のラベル画像情報）
                y_real_label = targets.to( self._device )
                y_real_one_hot = eye_tsr[y_real_label].view( -1, self._n_classes, 1, 1 ).to( self._device )
                y_real_image_label = y_real_one_hot.expand( self._batch_size, self._n_classes, images.shape[2], images.shape[3] ).to( self._device )

                #====================================================
                # 識別器 D の fitting 処理
                #====================================================
                # 生成器 G に入力するノイズ z (62 : ノイズの次元)
                # Generatorの更新の前にノイズを新しく生成しなおす必要があり。
                input_noize_z = torch.rand( size = (self._batch_size, self._n_input_noize_z) ).to( self._device )

                # 生成器に入力するクラスラベル y（＝偽のラベル情報）
                # 識別器と生成器の更新の前にノイズを新しく生成しなおす。
                y_fake_label = torch.randint( self._n_classes, (self._batch_size,), dtype = torch.long ).to( self._device )
                y_fake_one_hot = eye_tsr[y_fake_label].view( -1, self._n_classes ).to( self._device )

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
                D_x = self._dicriminator( images, y_real_image_label )
                #print( "D_x.size() :", D_x.size() )
                #print( "D_x :", D_x )

                # G(z) : 生成器から出力される偽物画像
                G_z = self._generator( input_noize_z, y_fake_one_hot )
                #print( "G_z.size() :", G_z.size() )
                #print( "G_z :", G_z )

                # D( G(z) ) : 偽物画像を入力したときの識別器の出力 (0.0 ~ 1.0)
                D_G_z = self._dicriminator( G_z, y_real_image_label )
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
                # Generatorの更新の前にノイズを新しく生成しなおす必要があり。
                input_noize_z = torch.rand( size = (self._batch_size, self._n_input_noize_z) ).to( self._device )

                # 生成器に入力するクラスラベル y（＝偽のラベル情報）
                # 生成器の更新の前にも、ノイズを新しく生成しなおす。
                y_fake_label = torch.randint( self._n_classes, (self._batch_size,), dtype = torch.long ).to( self._device )
                y_fake_one_hot = eye_tsr[y_fake_label].view( -1, self._n_classes ).to( self._device )

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
                G_z = self._generator( input_noize_z, y_fake_one_hot )
                #print( "G_z.size() :", G_z.size() )
                #print( "G_z :", G_z )

                # D( G(z) ) : 偽物画像を入力したときの識別器の出力 (0.0 ~ 1.0)
                D_G_z = self._dicriminator( G_z, y_real_image_label )
                #print( "D_G_z.size() :", D_G_z.size() )

                #----------------------------------------------------
                # 損失関数を計算する
                #----------------------------------------------------
                # L_G = E[ log{D(G(z))} ]
                loss_G = self._loss_fn( D_G_z, ones_tsr )
                #print( "loss_G :", loss_G )
                self._loss_G_historys.append( loss_G.item() )

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
            # 特定のエポックでGeneratorから画像を保存
            if( epoch % n_sava_step == 0 ):
                images = self.generate_fixed_images( b_transformed = False )
                self._images_historys.append( images )
                save_image( tensor = images, filename = "CGANforMNIST_Image_epoches{}_iters{}.png".format( epoch, iterations ) )

                for i in range( self._n_classes ):
                    images_i = self.generate_fixed_images_with_lable( y_label = i, b_transformed = False )
                    save_image( tensor = images_i, filename = "CGANforMNIST_Image{}_epoches{}_iters{}.png".format( i, epoch, iterations ) )


        print("Finished Training Loop.")
        return


    def generate_images( self, n_samples = 64, b_transformed = False ):
        """
        GAN の Generator から、画像データを自動生成する。
        [Input]
            n_samples : <int> 生成する画像の枚数
            b_transformed : <bool> 画像のフォーマットを Tensor から変換するか否か
        [Output]
            images : <Tensor> / shape = [n_samples, n_channels, height, width]
                生成された画像データのリスト
                行成分は生成する画像の数 n_samples
        """
        # 生成器を推論モードに切り替える。
        self._generator.eval()

        # 生成のもとになる乱数を生成
        input_noize_z = torch.rand( (n_samples, self._n_input_noize_z) ).to( self._device )

        # 生成器に入力するクラスラベル y
        eye_tsr = torch.eye( self._n_classes ).to( self._device )
        y_fake_label = torch.randint( self._n_classes, (n_samples,), dtype = torch.long ).to( self._device )
        y_fake_one_hot = eye_tsr[y_fake_label].view( -1, self._n_classes ).to( self._device )

        # 画像を生成
        images = self._generator( input_noize_z, y_fake_one_hot )
        #print( "images.size() :", images.size() )   # torch.Size([64, 1, 28, 28])

        if( b_transformed == True ):
            # Tensor → numpy に変換
            images = images.cpu().detach().numpy()

        return images


    def generate_images_with_lable( self, n_samples = 64, y_label = 0, b_transformed = False ):
        """
        引数で指定したラベルの画像を自動生成する。
        [Args]
            n_samples : <int> 生成する画像の枚数
            y_label : <int> 生成したい画像のクラスラベル
            b_transformed : <bool> 画像のフォーマットを Tensor から変換するか否か
        [Returns]
            images : <Tensor> / shape = [n_samples, n_channels, height, width]
                生成された画像データのリスト
                行成分は生成する画像の数 n_samples
        """
        # 生成器を推論モードに切り替える。
        self._generator.eval()

        # 生成のもとになる乱数を生成
        input_noize_z = torch.rand( (n_samples, self._n_input_noize_z) ).to( self._device )

        # 生成器に入力するクラスラベル y
        eye_tsr = torch.eye( self._n_classes ).to( self._device )
        y_fake_label = torch.full( (n_samples,), y_label ).long().to( self._device )
        y_fake_one_hot = eye_tsr[y_fake_label].view( -1, self._n_classes ).to( self._device )

        # 画像を生成
        images = self._generator( input_noize_z, y_fake_one_hot )
        #print( "images.size() :", images.size() )   # torch.Size([64, 1, 28, 28])

        if( b_transformed == True ):
            # Tensor → numpy に変換
            images = images.cpu().detach().numpy()

        return images

    def generate_fixed_images( self, n_samples = 64, b_transformed = False ):
        """
        CGAN の Generator から、固定された画像データを自動生成する。
        [Args]
            n_samples : <int> 生成する画像の枚数
            b_transformed : <bool> 画像のフォーマットを Tensor から変換するか否か
        [Returns]
            images : <Tensor> / shape = [n_samples, n_channels, height, width]
                生成された画像データのリスト
                行成分は生成する画像の数 n_samples
        """
        # 生成器を推論モードに切り替える。
        self._generator.eval()

        # 生成器に入力するクラスラベル y
        eye_tsr = torch.eye( self._n_classes ).to( self._device )
        y_fake_label = torch.randint( self._n_classes, (n_samples,), dtype = torch.long ).to( self._device )
        y_fake_one_hot = eye_tsr[y_fake_label].view( -1, self._n_classes ).to( self._device )

        # 画像を生成
        images = self._generator( self._fixed_input_noize_z, y_fake_one_hot )
        #print( "images.size() :", images.size() )

        if( b_transformed == True ):
            # Tensor → numpy に変換
            images = images.cpu().detach().numpy()

        return images

    def generate_fixed_images_with_lable( self, n_samples = 64, y_label = 0, b_transformed = False ):
        """
        引数で指定したラベルの画像を自動生成する。
        [Args]
            n_samples : <int> 生成する画像の枚数
            y_label : <int> 生成したい画像のクラスラベル
            b_transformed : <bool> 画像のフォーマットを Tensor から変換するか否か
        [Returns]
            images : <Tensor> / shape = [n_samples, n_channels, height, width]
                生成された画像データのリスト
                行成分は生成する画像の数 n_samples
        """
        # 生成器を推論モードに切り替える。
        self._generator.eval()

        # 生成器に入力するクラスラベル y
        eye_tsr = torch.eye( self._n_classes ).to( self._device )
        y_fake_label = torch.full( (n_samples,), y_label ).long().to( self._device )
        y_fake_one_hot = eye_tsr[y_fake_label].view( -1, self._n_classes ).to( self._device )

        # 画像を生成
        images = self._generator( self._fixed_input_noize_z, y_fake_one_hot )
        #print( "images.size() :", images.size() )

        if( b_transformed == True ):
            # Tensor → numpy に変換
            images = images.cpu().detach().numpy()

        return images
