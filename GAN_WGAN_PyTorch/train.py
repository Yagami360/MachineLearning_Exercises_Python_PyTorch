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
from networks import Generator, Discriminator, PatchGANDiscriminator
from utils import save_checkpoint, load_checkpoint
from utils import board_add_image, board_add_images
from utils import save_image_historys_gif

if __name__ == '__main__':
    """
    WGAN による学習処理
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="WGAN_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset', choices=['mnist', 'cifar-10'], default="mnist", help="データセットの種類（MNIST or CIFAR-10）")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="データセットのディレクトリ")
    parser.add_argument('--results_dir', type=str, default="results", help="生成画像の出力ディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_dir', type=str, default="", help="モデルの読み込みディレクトリ")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--n_test', type=int, default=10000, help="test dataset の最大数")
    parser.add_argument('--n_epoches', type=int, default=100, help="エポック数")
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=4, help="test データのバッチサイズ")
    parser.add_argument('--optimizer', choices=['RMSprop','Adam' ], default="RMSprop", help="最適化アルゴリズムの種類")
    parser.add_argument('--lr_G', type=float, default=0.00005, help="学習率")
    parser.add_argument('--lr_D', type=float, default=0.00005, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_size', type=int, default=64, help="入力画像のサイズ（pixel単位）")
    parser.add_argument('--n_fmaps', type=int, default=64, help="特徴マップの枚数")
    parser.add_argument('--n_input_noize_z', type=int, default=100, help="生成器に入力するノイズ z の次数")
    parser.add_argument('--networkD_type', choices=['vanilla','PatchGAN' ], default="vanilla", help="GAN の識別器の種類")
    parser.add_argument('--n_critic', type=int, default=5, help="クリティックの更新回数")
    parser.add_argument('--w_clamp_upper', type=float, default=0.01, help="重みクリッピングの下限値")
    parser.add_argument('--w_clamp_lower', type=float, default=-0.01, help="重みクリッピングの下限値")
    parser.add_argument('--n_display_step', type=int, default=100, help="tensorboard への表示間隔")
    parser.add_argument('--n_display_test_step', type=int, default=1000, help="test データの tensorboard への表示間隔")
    parser.add_argument("--n_save_step", type=int, default=5000, help="モデルのチェックポイントの保存間隔")
    parser.add_argument("--seed", type=int, default=8, help="乱数シード値")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
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

    # 各種出力ディレクトリ
    if not( os.path.exists(args.results_dir) ):
        os.mkdir(args.results_dir)
    if not( os.path.exists(os.path.join(args.results_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.results_dir, args.exper_name) )
    if not( os.path.exists(args.tensorboard_dir) ):
        os.mkdir(args.tensorboard_dir)
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name, "G")) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name, "G") )
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name, "D")) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name, "D") )

    # for visualation
    board_train = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
    board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_test") )

    # seed 値の固定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        raise NotImplementedError('dataset %s not implemented' % args.dataset)

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
    # Genrator
    if( args.dataset == "mnist" ):
        model_G = Generator( 
            n_input_noize_z = args.n_input_noize_z,
            n_channels = 1,         # グレースケールのチャンネル数 1
            n_fmaps = args.n_fmaps
        ).to( device )
    else:
        model_G = Generator( 
            n_input_noize_z = args.n_input_noize_z,
            n_channels = 3,         # RGBのチャンネル数 3
            n_fmaps = args.n_fmaps
        ).to( device )

    # Discriminator
    if( args.dataset == "mnist" ):
        if( args.networkD_type == "PatchGAN" ):
            model_D = PatchGANDiscriminator( 
                n_in_channels = 1,
                n_fmaps = args.n_fmaps
            ).to( device )
        else:
            model_D = Discriminator( 
                n_channels = 1, 
                n_fmaps = args.n_fmaps
            ).to( device )
    else:
        if( args.networkD_type == "PatchGAN" ):
            model_D = PatchGANDiscriminator( 
                n_in_channels = 3,
                n_fmaps = args.n_fmaps
            ).to( device )
        else:
            model_D = Discriminator( 
                n_channels = 3, 
                n_fmaps = args.n_fmaps
            ).to( device )
        
    if( args.debug ):
        print( "model_G :\n", model_G )
        print( "model_D :\n", model_D )

    # モデルを読み込む
    if not args.load_checkpoints_dir == '' and os.path.exists(args.load_checkpoints_dir):
        init_step = load_checkpoint(model_G, os.path.join(args.load_checkpoints_dir, "G") )
        init_step = load_checkpoint(model_D, os.path.join(args.load_checkpoints_dir, "D") )

    #======================================================================
    # optimizer の設定
    #======================================================================
    if( args.optimizer == "RMSprop" ):
        optimizer_G = optim.RMSprop(
            params = model_G.parameters(), lr = args.lr_G
        )
        
        optimizer_D = optim.RMSprop(
            params = model_D.parameters(), lr = args.lr_D
        )
    elif( args.optimizer == "Adam" ):
        optimizer_G = optim.Adam(
            params = model_G.parameters(),
            lr = args.lr_G, betas = (args.beta1,args.beta2)
        )
        
        optimizer_D = optim.Adam(
            params = model_D.parameters(),
            lr = args.lr_D, betas = (args.beta1,args.beta2)
        )
    else:
        raise NotImplementedError('optimizer %s not implemented' % args.optimizer)

    #======================================================================
    # loss 関数の設定
    #======================================================================
    pass

    #======================================================================
    # モデルの学習処理
    #======================================================================
    # 入力ノイズ z
    input_noize_z = torch.randn( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )
    input_noize_fix_z = torch.randn( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )
    input_noize_fix_z_test = torch.randn( size = (args.batch_size_test, args.n_input_noize_z,1,1) ).to( device )
    
    if( args.debug ):
        print( "input_noize_z.shape :", input_noize_z.shape )

    # 学習中の生成画像の履歴
    fake_images_historys = []

    print("Starting Training Loop...")
    iterations = 0      # 学習処理のイテレーション回数
    n_print = 1
    #-----------------------------
    # エポック数分トレーニング
    #-----------------------------
    for epoch in tqdm( range(args.n_epoches), desc = "Epoches" ):
        # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
        for step, (images,targets) in enumerate( tqdm( dloader_train, desc = "minbatch iters" ) ):
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

                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
                #----------------------------------------------------
                # 生成器 G に入力するノイズ z
                # Generatorの更新の前にノイズを新しく生成しなおす必要があり。
                input_noize_z = torch.rand( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )

                # E[C(x)] : 本物画像 x = image を入力したときのクリティックの出力 （平均化処理済み）
                C_x = model_D( images )
                if( args.debug and n_print > 0 ):
                    print( "C_x.size() :", C_x.size() )

                with torch.no_grad():   # 生成器 G の更新が行われないようにする。
                    # G(z) : 生成器から出力される偽物画像
                    G_z = model_G( input_noize_z )
                
                if( args.debug and n_print > 0 ):
                    print( "G_z.size() :", G_z.size() )     # torch.Size([128, 1, 28, 28])

                # E[ C( G(z) ) ] : 偽物画像を入力したときの識別器の出力 (平均化処理済み)
                C_G_z = model_D( G_z.detach()  )    # detach して G_z を通じて、生成器に勾配が伝搬しないようにする
                if( args.debug and n_print > 0 ):
                    print( "C_G_z.size() :", C_G_z.size() )

                #----------------------------------------------------
                # 損失関数を計算する
                # 出力と教師データを損失関数に設定し、誤差 loss を計算
                # この設定は、損失関数を __call__ をオーバライト
                # loss は Pytorch の Variable として帰ってくるので、これをloss.data[0]で数値として見る必要があり
                #----------------------------------------------------
                # E_x[ C(x) ]
                loss_C_real = torch.mean( C_x )

                # E_z[ C(G(z) ]
                loss_C_fake = torch.mean( C_G_z )

                # クリティック C の損失関数 = E_x[ C(x) ] + E_z[ C(G(z) ]
                loss_C = - loss_C_real + loss_C_fake

                #----------------------------------------------------
                # ネットワークの更新処理
                #----------------------------------------------------
                # 勾配を 0 に初期化（この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
                optimizer_D.zero_grad()

                # 勾配計算
                #loss_C.backward(retain_graph=True)
                loss_C.backward()

                # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
                optimizer_D.step()

            #====================================================
            # 生成器 G の fitting 処理
            #====================================================
            # クリティック C のネットワークの勾配計算を行わないようにする。
            for param in model_D.parameters():
                param.requires_grad = False

            #----------------------------------------------------
            # 学習用データをモデルに流し込む
            # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
            #----------------------------------------------------
            # 生成器 G に入力するノイズ z
            # Generatorの更新の前にノイズを新しく生成しなおす必要があり。
            input_noize_z = torch.rand( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )

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
            loss_G = - torch.mean( C_G_z )

            #----------------------------------------------------
            # ネットワークの更新処理
            #----------------------------------------------------
            # 勾配を 0 に初期化（この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
            optimizer_G.zero_grad()

            # 勾配計算
            loss_G.backward()

            # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
            optimizer_G.step()

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_display_step == 0 ) ):
                board_train.add_scalar('Generater/loss_G', loss_G.item(), iterations)
                board_train.add_scalar('Discriminator/loss_D', loss_C.item(), iterations)
                board_train.add_scalar('Discriminator/loss_D_real', loss_C_real.item(), iterations)
                board_train.add_scalar('Discriminator/loss_D_fake', loss_C_fake.item(), iterations)
                board_add_image(board_train, 'fake_image', G_z, iterations)
                print( "epoch={}, iters={}, loss_G={:.5f}, loss_C={:.5f}".format(epoch, iterations, loss_G, loss_C) )

            #====================================================
            # test loss の表示
            #====================================================
            if( step == 0 or ( step % args.n_display_test_step == 0 ) ):
                model_G.eval()
                model_D.eval()

                loss_C_real_total = 0
                loss_C_fake_total = 0
                loss_C_total = 0
                loss_G_total = 0
                n_test_loop = 0
                test_iterations = 0
                for (test_images,test_targets) in dloader_test :
                    if test_images.size()[0] != args.batch_size_test:
                        break

                    test_iterations += args.batch_size_test
                    n_test_loop += 1

                    #----------------------------------------------------
                    # 入力データをセット
                    #----------------------------------------------------
                    test_images = test_images.to( device )

                    #----------------------------------------------------
                    # テスト用データをモデルに流し込む
                    #----------------------------------------------------
                    with torch.no_grad():
                        C_x = model_D( test_images )
                        G_z = model_G( input_noize_fix_z_test )
                        C_G_z = model_D( G_z )

                    #----------------------------------------------------
                    # 損失関数を計算する
                    #----------------------------------------------------
                    test_loss_C_real = torch.mean( C_x )
                    test_loss_C_fake = torch.mean( C_G_z )
                    test_loss_C = - test_loss_C_real + test_loss_C_fake
                    test_loss_G = - torch.mean( C_G_z )

                    loss_C_real_total += test_loss_C_real.item()
                    loss_C_fake_total += test_loss_C_fake.item()
                    loss_C_total += test_loss_C.item()
                    loss_G_total += test_loss_G.item()

                    if( n_test_loop > args.n_test ):
                        break

                board_test.add_scalar('Generater/loss_G', loss_G_total/n_test_loop, iterations)
                board_test.add_scalar('Discriminator/loss_D', loss_C_total/n_test_loop, iterations)
                board_test.add_scalar('Discriminator/loss_D_real', loss_C_real_total/n_test_loop, iterations)
                board_test.add_scalar('Discriminator/loss_D_fake', loss_C_fake_total/n_test_loop, iterations)
                board_add_image(board_test, 'fake_image_test', G_z, iterations)

            #====================================================
            # モデルの保存
            #====================================================
            if( ( step % args.n_save_step == 0 ) ):
                #save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "G", 'step_%08d.pth' % (iterations + 1)), iterations )
                save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "G", 'G_final.pth'), iterations )
                #save_checkpoint( model_D, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "D", 'step_%08d.pth' % (iterations + 1)), iterations )
                save_checkpoint( model_D, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "D", 'D_final.pth'), iterations )
                print( "saved checkpoints" )

            n_print -= 1

        #====================================================
        # 各 Epoch 終了後の処理
        #====================================================
        # 出力画像の生成＆保存
        model_G.eval()
        with torch.no_grad():
            G_z = model_G( input_noize_fix_z )

        save_image( tensor = G_z[0], filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}_batch0.png".format( epoch ) )
        save_image( tensor = G_z, filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}_batchAll.png".format( epoch ) )

        fake_images_historys.append(G_z[0].transpose(0,1).transpose(1,2).cpu().clone().numpy())
        save_image_historys_gif( fake_images_historys, os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}.gif".format( epoch ) )        

    save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "G", 'G_final.pth'), iterations )
    save_checkpoint( model_D, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "D", 'D_final.pth'), iterations )
    print("Finished Training Loop.")
