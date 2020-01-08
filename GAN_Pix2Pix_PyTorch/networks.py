# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#====================================
# Generators
#====================================
class Generator( nn.Module ):
    """
    生成器 G [Generator] 側のネットワーク構成を記述したモデル。
    """
    def __init__(
        self,
        n_input_noize_z = 100,
        n_channels = 3,
        n_fmaps = 64
    ):
        super( Generator, self ).__init__()
        
        self.layer = nn.Sequential(
            # layer1
            nn.ConvTranspose2d(n_input_noize_z, n_fmaps*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_fmaps*8),
            nn.ReLU(inplace=True),

            # layer2
            nn.ConvTranspose2d( n_fmaps*8, n_fmaps*4, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps*4),
            nn.ReLU(inplace=True),

            # layer3
            nn.ConvTranspose2d( n_fmaps*4, n_fmaps*2, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps*2),
            nn.ReLU(inplace=True),

            # layer4
            nn.ConvTranspose2d( n_fmaps*2, n_fmaps, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.BatchNorm2d(n_fmaps),
            nn.ReLU(inplace=True),

            # output layer
            nn.ConvTranspose2d( n_fmaps, n_channels, kernel_size=4, stride=2, padding=1, bias=False ),
            nn.Tanh()
        )

        #weights_init( self )
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
        output = self.layer(input)
        return output


class Pix2PixUNetGenerator( nn.Module ):
    """
    UNet 構造での生成器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_out_channels = 3,
        n_fmaps = 64,
        dropout = 0.5   # 生成器 G に入力する入力ノイズ z は、直接 dropout を施すという意味でのノイズとして実現する。
    ):
        super( Pix2PixUNetGenerator, self ).__init__()

        def conv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True ),

                nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.Dropout( dropout )
            )
            return model

        def dconv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.ConvTranspose2d( in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1 ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU( 0.2, inplace=True ),
                nn.Dropout( dropout )
            )
            return model

        # Encoder（ダウンサンプリング）
        self.conv1 = conv_block( n_in_channels, n_fmaps )
        self.pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv2 = conv_block( n_fmaps*1, n_fmaps*2 )
        self.pool2 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv3 = conv_block( n_fmaps*2, n_fmaps*4 )
        self.pool3 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv4 = conv_block( n_fmaps*4, n_fmaps*8 )
        self.pool4 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )

        #
        self.bridge=conv_block( n_fmaps*8, n_fmaps*16 )

        # Decoder（アップサンプリング）
        self.dconv1 = dconv_block( n_fmaps*16, n_fmaps*8 )
        self.up1 = conv_block( n_fmaps*16, n_fmaps*8 )
        self.dconv2 = dconv_block( n_fmaps*8, n_fmaps*4 )
        self.up2 = conv_block( n_fmaps*8, n_fmaps*4 )
        self.dconv3 = dconv_block( n_fmaps*4, n_fmaps*2 )
        self.up3 = conv_block( n_fmaps*4, n_fmaps*2 )
        self.dconv4 = dconv_block( n_fmaps*2, n_fmaps*1 )
        self.up4 = conv_block( n_fmaps*2, n_fmaps*1 )

        # 出力層
        self.out_layer = nn.Sequential(
		    nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),
		    nn.Tanh(),
		)
        return

    def forward( self, input ):
        # Encoder（ダウンサンプリング）
        conv1 = self.conv1( input )
        pool1 = self.pool1( conv1 )
        conv2 = self.conv2( pool1 )
        pool2 = self.pool2( conv2 )
        conv3 = self.conv3( pool2 )
        pool3 = self.pool3( conv3 )
        conv4 = self.conv4( pool3 )
        pool4 = self.pool4( conv4 )

        #
        bridge = self.bridge( pool4 )

        # Decoder（アップサンプリング）& skip connection
        dconv1 = self.dconv1(bridge)

        concat1 = torch.cat( [dconv1,conv4], dim=1 )
        up1 = self.up1(concat1)

        dconv2 = self.dconv2(up1)
        concat2 = torch.cat( [dconv2,conv3], dim=1 )

        up2 = self.up2(concat2)
        dconv3 = self.dconv3(up2)
        concat3 = torch.cat( [dconv3,conv2], dim=1 )

        up3 = self.up3(concat3)
        dconv4 = self.dconv4(up3)
        concat4 = torch.cat( [dconv4,conv1], dim=1 )

        up4 = self.up4(concat4)

        # 出力層
        output = self.out_layer( up4 )

        return output

#====================================
# Discriminators
#====================================
class Pix2PixDiscriminator( nn.Module ):
    """
    識別器側のネットワーク構成を記述したモデル。
    """
    def __init__(
       self,
       n_channels = 3,
       n_fmaps = 64
    ):
        super( Pix2PixDiscriminator, self ).__init__() 
               
        self.layer = nn.Sequential(
            nn.Conv2d(n_channels*2, n_fmaps, kernel_size=4, stride=2, padding=1, bias=False),
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

        #weights_init( self )        
        return

    def forward(self, x,y):
        output = torch.cat( [x, y], dim=1 )
        output = self.layer( output )
        return output.view(-1)


class Pix2PixPatchGANDiscriminator( nn.Module ):
    """
    PatchGAN の識別器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_fmaps = 32
    ):
        super( Pix2PixPatchGANDiscriminator, self ).__init__()

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

        self.layer1 = discriminator_block1( n_in_channels * 2, n_fmaps )
        self.layer2 = discriminator_block2( n_fmaps, n_fmaps*2 )
        self.layer3 = discriminator_block2( n_fmaps*2, n_fmaps*4 )
        self.layer4 = discriminator_block2( n_fmaps*4, n_fmaps*8 )

        self.output_layer = nn.Sequential(
            nn.ZeroPad2d( (1, 0, 1, 0) ),
            nn.Conv2d( n_fmaps*8, 1, 4, padding=1, bias=False )
        )


    def forward(self, x, y ):
        output = torch.cat( [x, y], dim=1 )
        output = self.layer1( output )
        output = self.layer2( output )
        output = self.layer3( output )
        output = self.layer4( output )
        output = self.output_layer( output )
        output = output.view(-1)
        return output