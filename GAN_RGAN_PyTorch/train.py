# -*- coding:utf-8 -*-
import argparse
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import torchvision      # 画像処理関連
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 自作クラス
from networks import Generator, Discriminator
from visualization import board_add_image, board_add_images

if __name__ == '__main__':
    """
    WGAN-GP による学習処理
    ・学習用データセットは、MNIST / CIFAR-10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="WGAN_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="データセットのディレクトリ")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--dataset', choices=['mnist', 'cifar-10'], default="mnist", help="データセットの種類（MNIST or CIFAR-10）")
    parser.add_argument('--n_test', type=int, default=100, help="test dataset の最大数")
    parser.add_argument('--n_epoches', type=int, default=10, help="エポック数")
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=4, help="test データのバッチサイズ")
    parser.add_argument('--lr', type=float, default=0.0001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_size', type=int, default=64, help="入力画像のサイズ（pixel単位）")
    parser.add_argument('--n_channels', type=int, default=1, help="入力画像のチャンネル数")
    parser.add_argument('--n_fmaps', type=int, default=64, help="特徴マップの枚数")
    parser.add_argument('--n_input_noize_z', type=int, default=100, help="生成器に入力するノイズ z の次数")
    parser.add_argument('--gan_type', choices=['RSGAN','RSGAN-GP','RaGAN','RaLSGAN','RaSGAN-GP' ], default="RSGAN", help="GAN の種類）")
    parser.add_argument('--n_critic', type=int, default=1, help="クリティックの更新回数")
    parser.add_argument('--lambda_wgangp', type=float, default=10.0, help="WAGAN-GP の勾配ペナルティー係数")
    parser.add_argument('--n_display_step', type=int, default=100, help="tensorboard への表示間隔")
    parser.add_argument('--n_display_test_step', type=int, default=1000, help="test データの tensorboard への表示間隔")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 実行条件の出力
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "----------------------------------------------" )
    print( "開始時間：", datetime.now() )
    print( "PyTorch version :", torch.__version__ )
    for key, value in vars(args).items():
        print('%s: %s' % (str(key), str(value)))

    # 実行 Device の設定
    if( args.device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            #torch.cuda.set_device(args.gpu_ids[0])
            print( "実行デバイス :", device)
            print( "GPU名 :", torch.cuda.get_device_name(device))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device)
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device)

    print('-------------- End ----------------------------')

    #
    if not( os.path.exists(args.tensorboard_dir) ):
        os.mkdir(args.tensorboard_dir)

    # for visualation
    board_train = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
    board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_test") )

    # seed 値の固定
    #np.random.seed(8)
    #torch.manual_seed(8)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    if( args.dataset == "mnist" ):
        # データをロードした後に行う各種前処理の関数を構成を指定する。
        transform = transforms.Compose(
            [
                #transforms.Resize(args.image_size, interpolation=Image.LANCZOS ),
                transforms.Resize(args.image_size),
                transforms.ToTensor(),   # Tensor に変換]
                transforms.Normalize((0.5,), (0.5,)),   # 1 channel 分
            ]
        )

        # data と label をセットにした TensorDataSet の作成
        ds_train = torchvision.datasets.MNIST(
            root = args.dataset_dir,
            train = True,
            transform = transform,      # transforms.Compose(...) で作った前処理の一連の流れ
            target_transform = None,    
            download = True,
        )

        ds_test = torchvision.datasets.MNIST(
            root = args.dataset_dir,
            train = False,
            transform = transform,
            target_transform = None,
            download = True
        )

    elif( args.dataset == "cifar-10" ):
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=Image.LANCZOS ),
                transforms.ToTensor(),   # Tensor に変換
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        ds_train = torchvision.datasets.CIFAR10(
            root = args.dataset_dir,
            train = True,
            transform = transform,      # transforms.Compose(...) で作った前処理の一連の流れ
            target_transform = None,    
            download = True
        )

        ds_test = torchvision.datasets.CIFAR10(
            root = args.dataset_dir,
            train = False,
            transform = transform,
            target_transform = None,
            download = True
        )
    else:
        print( "Error: Invalid dataset" )
        exit()

    # TensorDataset → DataLoader への変換
    dloader_train = DataLoader(
        dataset = ds_train,
        batch_size = args.batch_size,
        shuffle = True
    )

    dloader_test = DataLoader(
        dataset = ds_test,
        batch_size = args.batch_size_test,
        shuffle = False
    )
    
    print( "ds_train :\n", ds_train ) # MNIST : torch.Size([60000, 28, 28]) , CIFAR-10 : (50000, 32, 32, 3)
    print( "ds_test :\n", ds_test )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model_G = Generator( 
        n_input_noize_z = args.n_input_noize_z, n_channels = args.n_channels, n_fmaps = args.n_fmaps
    ).to( device )

    model_D = Discriminator( 
        n_channels = args.n_channels, n_fmaps = args.n_fmaps
    ).to( device )

    if( args.debug ):
        print( "model_G :\n", model_G )
        print( "model_D :\n", model_D )

    # optimizer の設定
    optimizer_G = optim.Adam(
        params = model_G.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )
    
    optimizer_D = optim.Adam(
        params = model_D.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )

    # loss 関数の設定
    if( args.gan_type in ["RSGAN", "RaGAN"] ):
        #loss_fn = nn.BCELoss()             # when use sigmoid in Discriminator
        loss_fn = nn.BCEWithLogitsLoss()    # when not use sigmoid in Discriminator
    elif( args.gan_type in ["RaLSGAN"] ):
        loss_fn = nn.MSELoss()
    elif( args.gan_type in ["RSGAN-GP", "RaSGAN-GP"] ):
        loss_fn = None
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    #======================================================================
    # モデルの学習処理
    #======================================================================
    model_G.train()
    model_D.train()

    # 入力ノイズ z
    input_noize_z = torch.rand( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )
    if( args.debug ):
        print( "input_noize_z.shape :", input_noize_z.shape )

    # real ラベルを 1、fake ラベルを 0 として定義
    real_ones_tsr =  torch.ones( args.batch_size ).to( device )
    fake_zeros_tsr =  torch.zeros( args.batch_size ).to( device )

    print("Starting Training Loop...")
    iterations = 0      # 学習処理のイテレーション回数
    n_print = 1
    # エポック数分トレーニング
    for epoch in tqdm( range(args.n_epoches), desc = "Epoches" ):
        # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
        for (images,targets) in tqdm( dloader_train, desc = "minbatch iters" ):
            model_G.train()
            model_D.train()

            # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する
            # （後の計算で、shape の不一致をおこすため）
            if images.size()[0] != args.batch_size:
                break

            iterations += args.batch_size

            # ミニバッチデータを GPU へ転送
            images = images.to( device )

            # 入力ノイズを再生成 
            input_noize_z = torch.rand( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )

            #====================================================
            # 識別器 D の fitting 処理
            #====================================================
            # 無効化していた識別器 D のネットワークの勾配計算を有効化。
            for param in model_D.parameters():
                param.requires_grad = True

            for n in range( args.n_critic ):
                #----------------------------------------------------
                # 勾配を 0 に初期化
                # （この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
                #----------------------------------------------------
                optimizer_D.zero_grad()

                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
                #----------------------------------------------------
                # D(x) : 本物画像 x = image を入力したときの識別器の出力
                D_x = model_D( images )
                if( args.debug and n_print > 0 ):
                    print( "D_x.size() :", D_x.size() )
                    #print( "D_x :", D_x )

                # G(z) : 生成器から出力される偽物画像
                with torch.no_grad():   # 生成器 G の更新が行われないようにする。
                    G_z = model_G( input_noize_z )
                    if( args.debug and n_print > 0 ):
                        print( "G_z.size() :", G_z.size() )     # torch.Size([128, 1, 28, 28])
                    
                # D( G(z) ) : 偽物画像を入力したときの識別器の出力
                D_G_z = model_D( G_z.detach()  )    # detach して勾配が伝搬しないようにする
                if( args.debug and n_print > 0 ):
                    print( "D_G_z.size() :", D_G_z.size() )
                    #print( "D_G_z :", D_G_z )

                #----------------------------------------------------
                # 損失関数を計算する
                # 出力と教師データを損失関数に設定し、誤差 loss を計算
                # この設定は、損失関数を __call__ をオーバライト
                # loss は Pytorch の Variable として帰ってくるので、これをloss.data[0]で数値として見る必要があり
                #----------------------------------------------------
                if( args.gan_type == "RSGAN" ):
                    loss_D = loss_fn( D_x - D_G_z, real_ones_tsr )
                else:
                    # vanilla GAN
                    loss_D = loss_fn( D_x, real_ones_tsr ) + loss_fn( D_G_z, fake_zeros_tsr )

                #----------------------------------------------------
                # 勾配計算
                #----------------------------------------------------
                loss_D.backward(retain_graph=True)
                #loss_D.backward()

                #----------------------------------------------------
                # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
                #----------------------------------------------------
                optimizer_D.step()

            #====================================================
            # 生成器 G の fitting 処理
            #====================================================
            # 識別器 D のネットワークの勾配計算を行わないようにする。
            for param in model_D.parameters():
                param.requires_grad = False

            #----------------------------------------------------
            # 勾配を 0 に初期化
            # （この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
            #----------------------------------------------------
            optimizer_G.zero_grad()

            #----------------------------------------------------
            # 学習用データをモデルに流し込む
            # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
            #----------------------------------------------------
            # G(z) : 生成器から出力される偽物画像
            G_z = model_G( input_noize_z )
            if( args.debug and n_print > 0 ):
                print( "G_z.size() :", G_z.size() )

            # D( G(z) ) : 偽物画像を入力したときの識別器の出力
            #with torch.no_grad():  # param.requires_grad = False しているので不要
            D_G_z = model_D( G_z )
            if( args.debug and n_print > 0 ):
                print( "D_G_z.size() :", D_G_z.size() )

            #----------------------------------------------------
            # 損失関数を計算する
            #----------------------------------------------------
            if( args.gan_type == "RSGAN" ):
                loss_G = loss_fn( D_G_z - D_x, real_ones_tsr )
            else:
                # vanilla GAN
                loss_G = loss_fn( D_G_z, real_ones_tsr )

            #----------------------------------------------------
            # 勾配計算
            #----------------------------------------------------
            loss_G.backward()

            #----------------------------------------------------
            # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
            #----------------------------------------------------
            optimizer_G.step()

            #----------------------------------------------------
            # 学習過程の表示
            #----------------------------------------------------
            if( iterations == args.batch_size or ( iterations % args.n_display_step == 0 ) ):
                board_train.add_scalar('Generater/loss_G', loss_G.item(), iterations)
                board_train.add_scalar('Discriminator/loss_D', loss_D.item(), iterations)
                if( args.gan_type in ["RSGAN-GP", "RaSGAN-GP"] ):
                    board_train.add_scalar('Discriminator/gradient_penalty', gradient_penalty_loss.item(), iterations)
                board_add_image(board_train, 'fake image', G_z, iterations+1)


            n_print -= 1

        print("Finished Training Loop.")