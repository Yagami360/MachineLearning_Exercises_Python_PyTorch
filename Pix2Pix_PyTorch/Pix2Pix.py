# -*- coding:utf-8 -*-

"""
    更新情報
    [19/05/01] : 新規作成
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


def weights_init( model ):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.zero_()
    return


class Generator( nn.Module ):
    """
    pix2pix の生成器 G [Generator] 側のネットワーク構成を記述したモデル。
    ・ネットワーク構成は UNet ベース

    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 使用デバイス
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
        self,
        device,
        n_in_channels = 3,
        n_out_channels = 3,
        n_fmaps = 64,
        dropout = 0.5
    ):
        """
        [Args]
            n_in_channels : <int> 入力画像のチャンネル数
            n_out_channels : <int> 出力画像のチャンネル数
            n_channels : <int> 特徴マップの枚数
        """
        super( Generator, self ).__init__()
        self._device = device

        def conv_block( in_dim, out_dim, dropout = 0.0 ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True ),

                nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),

                nn.Dropout( dropout )
            )
            return model

        def dconv_block( in_dim, out_dim, dropout = 0.0 ):
            model = nn.Sequential(
                nn.ConvTranspose2d( in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1 ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU( 0.2, inplace=True ),
                nn.Dropout( dropout )
            )
            return model

        # Encoder（ダウンサンプリング）
        self._conv1 = conv_block( n_in_channels, n_fmaps ).to( self._device )
        self._pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )
        self._conv2 = conv_block( n_fmaps*1, n_fmaps*2 ).to( self._device )
        self._pool2 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )
        self._conv3 = conv_block( n_fmaps*2, n_fmaps*4, dropout ).to( self._device )
        self._pool3 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )
        self._conv4 = conv_block( n_fmaps*4, n_fmaps*8, dropout ).to( self._device )
        self._pool4 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 ).to( self._device )

        #
        self._bridge=conv_block( n_fmaps*8, n_fmaps*16 ).to( self._device )

        # Decoder（アップサンプリング）
        self._dconv1 = dconv_block( n_fmaps*16, n_fmaps*8, dropout ).to( self._device )
        self._up1 = conv_block( n_fmaps*16, n_fmaps*8, dropout ).to( self._device )
        self._dconv2 = dconv_block( n_fmaps*8, n_fmaps*4, dropout ).to( self._device )
        self._up2 = conv_block( n_fmaps*8, n_fmaps*4, dropout ).to( self._device )
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



class Discriminator( nn.Module ):
    """
    Pix2Pix の識別器 D [Discriminator] 側のネットワーク構成を記述したモデル。
    ・ネットワーク構成は PatchGAN ベース
    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 使用デバイス
        _layer : <nn.Sequential> 識別器のネットワーク構成
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
       self,
       device,
       n_in_channels = 3,
       n_fmaps = 64
    ):
        super( Discriminator, self ).__init__()
        self._device = device

        # 識別器のネットワークでは、Patch GAN を採用するが、
        # patchを切り出したり、ストライドするような処理は、直接的には行わない
        # その代りに、これを畳み込みで表現する。
        # つまり、CNNを畳み込んで得られる特徴マップのある1pixelは、入力画像のある領域(Receptive field)の影響を受けた値になるが、
        # 裏を返せば、ある1pixelに影響を与えられるのは、入力画像のある領域だけ。
        # そのため、「最終出力をあるサイズをもった特徴マップにして、各pixelにて真偽判定をする」ことと 、「入力画像をpatchにして、各patchの出力で真偽判定をする」ということが等価になるためである。
        def discriminator_block1( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        def discriminator_block2( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.InstanceNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        self._layer1 = discriminator_block1( n_in_channels * 2, n_fmaps ).to( self._device )
        self._layer2 = discriminator_block2( n_fmaps, n_fmaps*2 ).to( self._device )
        self._layer3 = discriminator_block2( n_fmaps*2, n_fmaps*4 ).to( self._device )
        self._layer4 = discriminator_block2( n_fmaps*4, n_fmaps*8 ).to( self._device )
        self._output_layer = nn.Sequential(
            nn.ZeroPad2d( (1, 0, 1, 0) ),
            nn.Conv2d( n_fmaps*8, 1, 4, padding=1, bias=False )
        ).to( self._device )

        #weights_init( self )
        return

    def forward(self, x, y):
        """
        [Args]
            x : <Tensor> image-to-image 変換前の画像データ
            y : <Tensor> image-to-image 変換後の画像データ
        """
        output = torch.cat( [x, y], dim=1 )
        output = self._layer1( output )
        output = self._layer2( output )
        output = self._layer3( output )
        output = self._layer4( output )
        output = self._output_layer( output )
        output = output.squeeze()
        return output


class Pix2PixModel( object ):
    """
    Pix2Pix を表すクラス

    --------------------------------------------
    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 実行デバイス
        _n_epoches : <int> エポック数（学習回数）
        _learnig_rate : <float> 最適化アルゴリズムの学習率
        _batch_size : <int> ミニバッチ学習時のバッチサイズ
        _n_channels : <int> 入力画像のチャンネル数
        _n_fmaps : <int> 特徴マップの枚数
        _image_width : <int> 画像の幅サイズ
        _image_height : <int> 画像の縦サイズ

        _generator : <nn.Module> 生成器
        _discriminator : <nn.Module> 識別器
        _loss_fn_cGAN : cGANの損失関数
        _loss_fn_pixelwise : L1損失関数
        _G_optimizer : <torch.optim.Optimizer> 生成器の最適化アルゴリズム
        _D_optimizer : <torch.optim.Optimizer> 識別器の最適化アルゴリズム
        _loss_G_historys : <list> 生成器 G の損失関数値の履歴（イテレーション毎）
        _loss_D_historys : <list> 識別器 D の損失関数値の履歴（イテレーション毎）
        _images_historys : <list> 生成画像のリスト

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
        n_fmaps = 64,
        image_width = 256,
        image_height = 256
    ):
        self._device = device

        self._n_epoches = n_epoches
        self._learning_rate = learing_rate
        self._batch_size = batch_size
        self._n_channels = n_channels
        self._n_fmaps = n_fmaps
        self._image_width = image_width
        self._image_height = image_height

        self._generator = None
        self._dicriminator = None
        self._loss_fn_cGAN = None
        self._loss_fn_pixelwise = None
        self._G_optimizer = None
        self._D_optimizer = None
        self._loss_G_historys = []
        self._loss_D_historys = []
        self._images_historys = []
        self.model()
        self.loss()
        self.optimizer()

        self._fixed_pre_image = None
        return

    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "Pix2PixModel" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_n_epoches :", self._n_epoches )
        print( "_learning_rate :", self._learning_rate )
        print( "_batch_size :", self._batch_size )
        print( "_n_channels :", self._n_channels )
        print( "_n_fmaps :", self._n_fmaps )
        print( "_image_width :", self._image_width )
        print( "_image_height :", self._image_height )
        print( "_generator :", self._generator )
        print( "_dicriminator :", self._dicriminator )
        print( "_loss_fn_cGAN :", self._loss_fn_cGAN )
        print( "_loss_fn_pixelwise :", self._loss_fn_pixelwise )
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
        self._generator = Generator( 
            self._device, 
            n_in_channels = self._n_channels,
            n_out_channels = self._n_channels,
            n_fmaps = self._n_fmaps,
            dropout = 0.5
        )

        self._dicriminator = Discriminator( 
            self._device,
            n_in_channels = self._n_channels,
            n_fmaps = self._n_fmaps
        )

        return

    def loss( self ):
        """
        損失関数の設定を行う。
        [Args]
        [Returns]
        """
        # MSELoss
        self._loss_fn_cGAN = nn.MSELoss()

        # L1正則化項
        self._loss_fn_pixelwise = torch.nn.L1Loss()
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


    def fit( self, dloader, n_sava_step = 50 ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Args]
            dloader : <DataLoader> 学習用データセットの DataLoader
            n_sava_step : <int> 学習途中での生成画像の保存間隔（エポック単位）
        [Returns]
        """
        # image-to-image の変換前の固定された画像（学習後の画像生成の入力画像として利用）の初期化済みフラグ
        b_init_fixed_image = False

        # 教師信号（０⇒偽物、1⇒本物）
        # real ラベルを 1 としてそして fake ラベルを 0 として定義
        patch = ( self._batch_size, self._image_height// 2 ** 4, self._image_width // 2 ** 4 )
        ones_tsr =  torch.ones( patch ).to( self._device )
        zeros_tsr =  torch.zeros( patch ).to( self._device )

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
                #images = images.to( self._device )

                # 学習用データには、左側に衛星画像、右側に地図画像が入っているので、chunk で切り分ける
                # torch.chunk() : 渡したTensorを指定した個数に切り分ける。
                # pre_image : image-to-image 変換タスクでの変換前の画像
                # after_image : image-to-image 変換タスクでの変換後の画像
                pre_image, after_image = torch.chunk( images, chunks=2, dim=3 )
                pre_image = pre_image.to( self._device )
                after_image = after_image.to( self._device )

                #====================================================
                # 識別器 D の fitting 処理
                #====================================================
                # 無効化していた識別器 D のネットワークの勾配計算を有効化。
                for param in self._dicriminator.parameters():
                    param.requires_grad = True

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
                D_x = self._dicriminator( pre_image, after_image )
                #print( "D_x.size() :", D_x.size() )     # torch.Size([batch_size, 16, 16])
                #print( "D_x :", D_x )

                # G(z) : 生成器から出力される偽物画像
                G_z = self._generator( after_image )
                #print( "G_z.size() :", G_z.size() )
                #print( "G_z :", G_z )

                # D( G(z) ) : 偽物画像を入力したときの識別器の出力 (0.0 ~ 1.0)
                #D_G_z = self._dicriminator( G_z.detach(), after_image )
                D_G_z = self._dicriminator( G_z, after_image )
                #print( "D_G_z.size() :", D_G_z.size() )
                #print( "D_G_z :", D_G_z )

                #----------------------------------------------------
                # 損失関数を計算する
                # 出力と教師データを損失関数に設定し、誤差 loss を計算
                # この設定は、損失関数を __call__ をオーバライト
                # loss は Pytorch の Variable として帰ってくるので、これをloss.data[0]で数値として見る必要があり
                #----------------------------------------------------
                # cGAN での損失関数
                loss_D_real = self._loss_fn_cGAN( D_x, ones_tsr )
                #print( "loss_D_real : ", loss_D_real.item() )

                # L1正則化項の損失関数
                loss_D_fake = self._loss_fn_cGAN( D_G_z, zeros_tsr )
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
                # 識別器 D のネットワークの勾配計算を無効化。
                for param in self._dicriminator.parameters():
                    param.requires_grad = False

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
                G_z = self._generator( after_image )
                #print( "G_z.size() :", G_z.size() )
                #print( "G_z :", G_z )

                # D( G(z) ) : 偽物画像を入力したときの識別器の出力 (0.0 ~ 1.0)
                D_G_z = self._dicriminator( G_z, after_image )
                #print( "D_G_z.size() :", D_G_z.size() )

                #----------------------------------------------------
                # 損失関数を計算する
                #----------------------------------------------------
                # cGAN での損失関数
                loss_G_cGAN = self._loss_fn_cGAN( D_G_z, ones_tsr )
                #print( "loss_G_cGAN :", loss_G_cGAN )

                # L1正則化項
                loss_G_L1 = self._loss_fn_pixelwise( G_z, after_image )

                # 最終的な生成器の損失関数
                loss_L1_lamda = 100
                loss_G = loss_G_cGAN + loss_L1_lamda * loss_G_L1
                
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
                # 特定のイテレーションでGeneratorから画像を保存
                if( iterations % n_sava_step == 0 ):
                    """
                    if( b_init_fixed_image == False ):
                        b_init_fixed_image = True
                        # 初回ループの画像を、image-to-image の変換前の固定された画像（学習後の画像生成の入力画像として利用）として採用
                        self._fixed_pre_image = pre_image.detach()

                    image = self.generate_fixed_images( pre_image = self._fixed_pre_image, b_transformed = False )
                    save_image( tensor = image, filename = "Pix2Pix_Image_epoches{}_iters{}.png".format( epoch, iterations ) )

                    # [batc_size, n_channels, height, width] → [height, width, n_channels]
                    image_reshaped = image[0, :, :, :].squeeze().transpose( 0 ,2 )
                    self._images_historys.append( image_reshaped.cpu().detach().numpy() )
                    """
                    save_image( tensor = G_z.cpu(), filename = "Pix2Pix_Image_epoches{}_iters{}.png".format( epoch, iterations ) )

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
                    self._fixed_pre_image = pre_image.detach()

                image = self.generate_fixed_images( pre_image = self._fixed_pre_image, b_transformed = False )
                save_image( tensor = image, filename = "Pix2Pix_Image_epoches{}_iters{}.png".format( epoch, iterations ) )
                # [batc_size, n_channels, height, width] → [height, width, n_channels]
                image_reshaped = image[0, :, :, :].squeeze().transpose( 0 ,2 )
                self._images_historys.append( image_reshaped.cpu().detach().numpy() )
                """
                save_image( tensor = G_z.cpu(), filename = "Pix2Pix_Image_epoches{}_iters{}.png".format( epoch, iterations ) )

        #self.save_image_historys_gif( file_name = "Pix2Pix_Image_epoches{}_iters{}.gif".format( epoch, iterations ) )
        print("Finished Training Loop.")
        return


    def generate_fixed_images( self, pre_image, b_transformed = False ):
        """
        pix2pix の Generator から、固定された画像データを自動生成する。
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
        self._generator.eval()

        # 画像を生成
        images = self._generator( pre_image )

        if( b_transformed == True ):
            # Tensor → numpy に変換
            images = images.cpu().detach().numpy()

        return images

    def save_image_historys_gif( self, file_name = "Pix2Pix.gif" ):
        """
        学習中の生成画像の様子を gif ファイルで保存
        """
        import imageio
        imageio.mimsave( file_name, self._images_historys )
        return