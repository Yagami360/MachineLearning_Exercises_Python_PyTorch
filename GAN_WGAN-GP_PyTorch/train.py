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
from networks import Generator, Critic
from visualization import board_add_image, board_add_images

if __name__ == '__main__':
    """
    WGAN-gp による画像の自動生成
    ・学習用データセットは、MNIST / CIFAR-10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="WGAN_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="データセットのディレクトリ")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--dataset', choices=['mnist', 'cifar-10'], default="mnist", help="データセットの種類（MNIST or CIFAR-10）")
    parser.add_argument('--n_test', type=int, default=100, help="test dataset の最大数")
    parser.add_argument('--n_epoches', type=int, default=10, help="エポック数")
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=4, help="test データのバッチサイズ")
    parser.add_argument('--lr', type=float, default=0.00005, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_size', type=int, default=64, help="入力画像のサイズ（pixel単位）")
    parser.add_argument('--n_channels', type=int, default=1, help="入力画像のチャンネル数")
    parser.add_argument('--n_fmaps', type=int, default=64, help="特徴マップの枚数")
    parser.add_argument('--n_input_noize_z', type=int, default=100, help="生成器に入力するノイズ z の次数")
    parser.add_argument('--n_critic', type=int, default=5, help="クリティックの更新回数")
    parser.add_argument('--w_clamp_upper', type=float, default=0.01, help="重みクリッピングの下限値")
    parser.add_argument('--w_clamp_lower', type=float, default=-0.01, help="重みクリッピングの下限値")
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
    np.random.seed(8)
    torch.manual_seed(8)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    if( args.dataset == "mnist" ):
        # データをロードした後に行う各種前処理の関数を構成を指定する。
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=Image.LANCZOS ),
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

    model_D = Critic( 
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

    #======================================================================
    # モデルの学習処理
    #======================================================================
    model_G.train()
    model_D.train()

    # 入力ノイズ z
    input_noize_z = torch.FloatTensor( args.batch_size, args.n_input_noize_z ).to( device )

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

            #====================================================
            # クリティック C の fitting 処理
            #====================================================
            # 無効化していたクリティック C のネットワークの勾配計算を有効化。
            for param in model_D.parameters():
                param.requires_grad = True

            for n in range( args.n_critic ):
                #----------------------------------------------------
                # 重みクリッピング
                #----------------------------------------------------
                for param in model_D.parameters():
                    #print( "critic param :", param )
                    param.data.clamp_( args.w_clamp_lower, args.w_clamp_upper )
                    #print( "critic param :", param )

                # 生成器 G に入力するノイズ z
                # Generatorの更新の前にノイズを新しく生成しなおす必要があり。
                input_noize_z.resize_( args.batch_size, args.n_input_noize_z, 1 , 1 ).normal_(0, 1)

                #----------------------------------------------------
                # 勾配を 0 に初期化
                # （この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
                #----------------------------------------------------
                optimizer_D.zero_grad()

                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
                #----------------------------------------------------
                # E[C(x)] : 本物画像 x = image を入力したときのクリティックの出力 （平均化処理済み）
                C_x = model_D( images )
                if( args.debug and n_print > 0 ):
                    print( "C_x.size() :", C_x.size() )
                    #print( "C_x :", C_x )

                # G(z) : 生成器から出力される偽物画像
                G_z = model_G( input_noize_z )

                # 微分を行わない処理の範囲を with 構文で囲む
                # クリティック D の学習中は、生成器 G のネットワークの勾配は更新しない。
                #with torch.no_grad():
                    #G_z = model_G( input_noize_z )
                
                if( args.debug and n_print > 0 ):
                    print( "G_z.size() :", G_z.size() )     # torch.Size([128, 1, 28, 28])
                    #print( "G_z :", G_z )

                # E[ C( G(z) ) ] : 偽物画像を入力したときの識別器の出力 (平均化処理済み)
                C_G_z = model_D( G_z )
                if( args.debug and n_print > 0 ):
                    print( "C_G_z.size() :", C_G_z.size() )
                    #print( "C_G_z :", C_G_z )

                #----------------------------------------------------
                # 損失関数を計算する
                # 出力と教師データを損失関数に設定し、誤差 loss を計算
                # この設定は、損失関数を __call__ をオーバライト
                # loss は Pytorch の Variable として帰ってくるので、これをloss.data[0]で数値として見る必要があり
                #----------------------------------------------------
                # E_x[ C(x) ]
                #loss_C_real = torch.mean( C_x )
                loss_C_real = C_x
                if( args.debug and n_print > 0 ):
                    print( "loss_C_real : ", loss_C_real.item() )

                # E_z[ C(G(z) ]
                #loss_C_fake = torch.mean( C_G_z )
                loss_C_fake = C_G_z
                if( args.debug and n_print > 0 ):
                    print( "loss_C_fake : ", loss_C_fake.item() )

                # クリティック C の損失関数 = E_x[ C(x) ] + E_z[ C(G(z) ]
                loss_C = loss_C_real - loss_C_fake
                if( args.debug and n_print > 0 ):
                    print( "loss_C : ", loss_C.item() )

                #----------------------------------------------------
                # 誤差逆伝搬
                #----------------------------------------------------
                loss_C.backward()

                #----------------------------------------------------
                # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
                #----------------------------------------------------
                optimizer_D.step()

            #====================================================
            # 生成器 G の fitting 処理
            #====================================================
            # クリティック C のネットワークの勾配計算を行わないようにする。
            for param in model_D.parameters():
                param.requires_grad = False

            # 生成器 G に入力するノイズ z
            # Generatorの更新の前にノイズを新しく生成しなおす必要があり。
            input_noize_z.resize_( args.batch_size, args.n_input_noize_z, 1, 1 ).normal_(0, 1)

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
                #print( "G_z :", G_z )

            # E[C( G(z) )] : 偽物画像を入力したときのクリティックの出力（平均化処理済み）
            C_G_z = model_D( G_z )
            if( args.debug and n_print > 0 ):
                print( "C_G_z.size() :", C_G_z.size() )

            #----------------------------------------------------
            # 損失関数を計算する
            #----------------------------------------------------
            # L_G = E_z[ C(G(z) ]
            #loss_G = torch.mean( C_G_z )
            loss_G = C_G_z
            if( args.debug and n_print > 0 ):
                print( "loss_G :", loss_G )

            #----------------------------------------------------
            # 誤差逆伝搬
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
                board_train.add_scalar('Critic/loss_C', loss_C.item(), iterations)
                board_train.add_scalar('Critic/loss_C_real', loss_C_real.item(), iterations)
                board_train.add_scalar('Critic/loss_C_fake', loss_C_fake.item(), iterations)
                board_add_image(board_train, 'fake image', G_z, iterations+1)

            if( iterations == args.batch_size or ( iterations % args.n_display_test_step == 0 ) ):
                model_G.eval()
                model_D.eval()

                loss_C_real_total = 0
                loss_C_fake_total = 0
                loss_C_total = 0
                loss_G_total = 0
                #for i, test_data in enumerate( test_dataset ):
                n_test_loop = 0
                with torch.no_grad():
                    for (test_images,test_targets) in dloader_test :
                        test_images = test_images.to( device )
                        C_x = model_D( test_images )
                        G_z = model_G( input_noize_z )
                        C_G_z = model_D( G_z )

                        test_loss_C_real = C_x
                        test_loss_C_fake = C_G_z
                        test_loss_C = test_loss_C_real - test_loss_C_fake

                        input_noize_z.resize_( args.batch_size, args.n_input_noize_z, 1, 1 ).normal_(0, 1)
                        G_z = model_G( input_noize_z )
                        C_G_z = model_D( G_z )
                        test_loss_G = C_G_z

                        loss_C_real_total += test_loss_C_real.item()
                        loss_C_fake_total += test_loss_C_fake.item()
                        loss_C_total += test_loss_C.item()
                        loss_G_total += test_loss_G.item()

                        n_test_loop += 1
                        if( n_test_loop > args.n_test ):
                            break

                board_test.add_scalar('Generater/loss_G', loss_G_total/n_test_loop, iterations)
                board_test.add_scalar('Critic/loss_C', loss_C_total/n_test_loop, iterations)
                board_test.add_scalar('Critic/loss_C_real', loss_C_real_total/n_test_loop, iterations)
                board_test.add_scalar('Critic/loss_C_fake', loss_C_fake_total/n_test_loop, iterations)
                board_add_image(board_test, 'fake image', G_z, iterations+1)

            n_print -= 1

        #----------------------------------------------------
        # 学習過程の表示
        #----------------------------------------------------
        n_sava_step_epoch = 1
        # 特定のエポックでGeneratorから画像を保存
        if( epoch % n_sava_step_epoch == 0 ):
            board_add_image(board_train, 'fake image', G_z, iterations+1)
            board_train.add_scalar('Generater/loss_G', loss_G.item(), iterations)
            board_train.add_scalar('Critic/loss_C', loss_C.item(), iterations)
            board_train.add_scalar('Critic/loss_C_real', loss_C_real.item(), iterations)
            board_train.add_scalar('Critic/loss_C_fake', loss_C_fake.item(), iterations)

        print("Finished Training Loop.")
