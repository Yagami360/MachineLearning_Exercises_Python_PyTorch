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
from networks import Pix2PixUNetGenerator, Pix2PixDiscriminator, Pix2PixPatchGANDiscriminator
from map2aerial_dataset import Map2AerialDataset, Map2AerialDataLoader
from utils import save_checkpoint, load_checkpoint
from utils import board_add_image, board_add_images
from utils import save_image_historys_gif

if __name__ == '__main__':
    """
    Pix2Pix による学習処理
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="Pix2Pix_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset_dir', type=str, default="dataset/maps", help="データセットのディレクトリ")
    parser.add_argument('--results_dir', type=str, default="results", help="生成画像の出力ディレクトリ")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_dir', type=str, default="", help="モデルの読み込みディレクトリ")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--n_test', type=int, default=10000, help="test dataset の最大数")
    parser.add_argument('--n_epoches', type=int, default=100, help="エポック数")
    parser.add_argument('--batch_size', type=int, default=32, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=4, help="test データのバッチサイズ")
    parser.add_argument('--lr', type=float, default=0.0001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_size', type=int, default=64, help="入力画像のサイズ（pixel単位）")
    parser.add_argument('--lambda_l1', type=float, default=100.0, help="L1正則化項の係数値")
    parser.add_argument('--unetG_dropout', type=float, default=0.5, help="生成器への入力ノイズとしての Dropout 率")
    parser.add_argument('--n_fmaps', type=int, default=64, help="特徴マップの枚数")
    parser.add_argument('--networkD_type', choices=['vanilla','PatchGAN' ], default="PatchGAN", help="GAN の識別器の種類")
    parser.add_argument('--n_display_step', type=int, default=50, help="tensorboard への表示間隔")
    parser.add_argument('--n_display_test_step', type=int, default=500, help="test データの tensorboard への表示間隔")
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
    ds_train = Map2AerialDataset( args.dataset_dir, "train", args.image_size, args.image_size, args.debug )
    ds_test = Map2AerialDataset( args.dataset_dir, "val", args.image_size, args.image_size, args.debug )

    dloader_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True )
    dloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size_test, shuffle=False )
    #dloader_train = Map2AerialDataLoader(ds_train, batch_size=args.batch_size, shuffle=True )
    #dloader_test = Map2AerialDataLoader(ds_test, batch_size=args.batch_size_test, shuffle=False )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    # Genrator
    model_G = Pix2PixUNetGenerator( 
        n_in_channels = 3, n_out_channels = 3,
        n_fmaps = args.n_fmaps,
        dropout = args.unetG_dropout
    ).to( device )

    # Discriminator
    if( args.networkD_type == "PatchGAN" ):
        model_D = Pix2PixPatchGANDiscriminator( 
            n_in_channels = 3,
            n_fmaps = args.n_fmaps
        ).to( device )
    else:
        model_D = Pix2PixDiscriminator( 
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
    optimizer_G = optim.Adam(
        params = model_G.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )
    
    optimizer_D = optim.Adam(
        params = model_D.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )

    #======================================================================
    # loss 関数の設定
    #======================================================================
    #loss_gan_fn = nn.BCELoss()             # when use sigmoid in Discriminator
    loss_gan_fn = nn.BCEWithLogitsLoss()    # when not use sigmoid in Discriminator

    loss_l1_fn = torch.nn.L1Loss()

    #======================================================================
    # モデルの学習処理
    #======================================================================
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
        for step, inputs in enumerate( tqdm( dloader_train, desc = "minbatch iters" ) ):
            model_G.train()
            model_D.train()

            # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する
            # （後の計算で、shape の不一致をおこすため）
            if inputs["aerial_image_tsr"].shape[0] != args.batch_size:
                break

            iterations += args.batch_size

            # ミニバッチデータを GPU へ転送
            pre_image = inputs["aerial_image_tsr"].to(device)
            after_image = inputs["map_image_tsr"].to(device)
            #save_image( pre_image, "pre_image.png" )
            #save_image( after_image, "after_image.png" )

            #====================================================
            # 識別器 D の fitting 処理
            #====================================================
            # 無効化していた識別器 D のネットワークの勾配計算を有効化。
            for param in model_D.parameters():
                param.requires_grad = True

            #----------------------------------------------------
            # 学習用データをモデルに流し込む
            # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
            #----------------------------------------------------
            # D(x) : 本物画像 x = image を入力したときの識別器の出力
            D_x = model_D( pre_image, after_image )
            if( args.debug and n_print > 0 ):
                print( "D_x.size() :", D_x.size() )

            # G(z) : 生成器から出力される偽物画像
            with torch.no_grad():   # 生成器 G の更新が行われないようにする。
                G_z = model_G( after_image )
                if( args.debug and n_print > 0 ):
                    print( "G_z.size() :", G_z.size() )
                
            # D( G(z) ) : 偽物画像を入力したときの識別器の出力
            D_G_z = model_D( G_z.detach(), after_image )    # detach して G_z を通じて、生成器に勾配が伝搬しないようにする
            #D_G_z = model_D( G_z, after_image )
            if( args.debug and n_print > 0 ):
                print( "D_G_z.size() :", D_G_z.size() )

            #----------------------------------------------------
            # 損失関数を計算する
            # 出力と教師データを損失関数に設定し、誤差 loss を計算
            # この設定は、損失関数を __call__ をオーバライト
            # loss は Pytorch の Variable として帰ってくるので、これをloss.data[0]で数値として見る必要があり
            #----------------------------------------------------
            # real ラベルを 1、fake ラベルを 0 として定義
            real_ones_tsr =  torch.ones( D_x.shape ).to( device )
            fake_zeros_tsr =  torch.zeros( D_x.shape ).to( device )
            if( args.debug and n_print > 0 ):
                print( "real_ones_tsr.shape :", real_ones_tsr.shape )
                print( "fake_zeros_tsr.shape :", fake_zeros_tsr.shape )

            loss_D_real = loss_gan_fn( D_x, real_ones_tsr )
            loss_D_fake = loss_gan_fn( D_G_z, fake_zeros_tsr )
            loss_D = loss_D_real + loss_D_fake

            #----------------------------------------------------
            # ネットワークの更新処理
            #----------------------------------------------------
            # 勾配を 0 に初期化（この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
            optimizer_D.zero_grad()

            # 勾配計算
            #loss_D.backward(retain_graph=True)
            loss_D.backward()

            # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
            optimizer_D.step()

            #====================================================
            # 生成器 G の fitting 処理
            #====================================================
            # 識別器 D のネットワークの勾配計算を行わないようにする。
            for param in model_D.parameters():
                param.requires_grad = False

            #----------------------------------------------------
            # 学習用データをモデルに流し込む
            # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
            #----------------------------------------------------
            # G(z) : 生成器から出力される偽物画像
            G_z = model_G( after_image )
            if( args.debug and n_print > 0 ):
                print( "G_z.size() :", G_z.size() )

            # D( G(z) ) : 偽物画像を入力したときの識別器の出力
            #with torch.no_grad():  # param.requires_grad = False しているので不要
            D_G_z = model_D( G_z, after_image )
            if( args.debug and n_print > 0 ):
                print( "D_G_z.size() :", D_G_z.size() )

            #----------------------------------------------------
            # 損失関数を計算する
            #----------------------------------------------------
            loss_gan = loss_gan_fn( D_G_z, real_ones_tsr )
            loss_l1 = loss_l1_fn( G_z, after_image )
            loss_G = loss_gan + args.lambda_l1 * loss_l1

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
                board_train.add_scalar('Generater/loss_gan', loss_gan.item(), iterations)
                board_train.add_scalar('Generater/loss_l1', loss_l1.item(), iterations)
                board_train.add_scalar('Discriminator/loss_D', loss_D.item(), iterations)
                board_train.add_scalar('Discriminator/loss_D_real', loss_D_real.item(), iterations)
                board_train.add_scalar('Discriminator/loss_D_fake', loss_D_fake.item(), iterations)
                print( "epoch={}, iters={}, loss_G={:.5f}, loss_D={:.5f}".format(epoch, iterations, loss_G, loss_D) )

                visuals = [
                    [pre_image, after_image, G_z],
                ]
                board_add_images(board_train, 'fake image', visuals, iterations+1)

            #====================================================
            # test loss の表示
            #====================================================
            if( step == 0 or ( step % args.n_display_test_step == 0 ) ):
                model_G.eval()
                model_D.eval()

                n_test_loop = 0
                test_iterations = 0
                loss_D_total = 0
                loss_D_real_total = 0
                loss_D_fake_total = 0
                loss_G_total = 0
                loss_gan_total = 0
                loss_l1_total = 0
                for test_inputs in dloader_test :
                    if test_inputs["aerial_image_tsr"].shape[0] != args.batch_size_test:
                        break

                    test_iterations += args.batch_size_test
                    n_test_loop += 1

                    #----------------------------------------------------
                    # 入力データをセット
                    #----------------------------------------------------
                    pre_image = test_inputs["aerial_image_tsr"].to(device)
                    after_image = test_inputs["map_image_tsr"].to(device)

                    #----------------------------------------------------
                    # テスト用データをモデルに流し込む
                    #----------------------------------------------------
                    with torch.no_grad():
                        D_x = model_D( pre_image, after_image )
                        G_z = model_G( after_image )
                        D_G_z = model_D( G_z, after_image )

                    #----------------------------------------------------
                    # 損失関数を計算する
                    #----------------------------------------------------
                    real_ones_tsr =  torch.ones( D_x.shape ).to( device )
                    fake_zeros_tsr =  torch.zeros( D_x.shape ).to( device )

                    # Discriminator
                    loss_D_real = loss_gan_fn( D_x, real_ones_tsr )
                    loss_D_fake = loss_gan_fn( D_G_z, fake_zeros_tsr )
                    loss_D = loss_D_real + loss_D_fake

                    # Generator
                    loss_gan = loss_gan_fn( D_G_z, real_ones_tsr )
                    loss_l1 = loss_l1_fn( G_z, after_image )
                    loss_G = loss_gan + args.lambda_l1 * loss_l1

                    # total
                    loss_D_total += loss_D.item()
                    loss_D_real_total += loss_D_real.item()
                    loss_D_fake_total += loss_D_fake.item()
                    loss_G_total += loss_G.item()
                    loss_gan_total += loss_gan.item()
                    loss_l1_total += loss_l1.item()

                    if( test_iterations > args.n_test ):
                        break

                board_test.add_scalar('Generater/loss_G', (loss_G_total/n_test_loop), iterations)
                board_test.add_scalar('Generater/loss_gan', (loss_gan_total/n_test_loop), iterations)
                board_test.add_scalar('Generater/loss_l1', (loss_l1_total/n_test_loop), iterations)
                board_test.add_scalar('Discriminator/loss_D', (loss_D_total/n_test_loop), iterations)
                board_test.add_scalar('Discriminator/loss_D_real', (loss_D_real_total/n_test_loop), iterations)
                board_test.add_scalar('Discriminator/loss_D_fake', (loss_D_fake_total/n_test_loop), iterations)

                visuals = [
                    [pre_image, after_image, G_z],
                ]
                """
                if( args.debug and n_print > 0 ):
                    for col, vis_item_row in enumerate(visuals):
                        for row, vis_item in enumerate(vis_item_row):
                            print("[test] vis_item[{}][{}].shape={} :".format(row,col,vis_item.shape) )
                """
                board_add_images(board_test, "fake image test", visuals, iterations)

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
        for test_inputs in dloader_test :
            fix_pre_image = test_inputs["aerial_image_tsr"].to(device)
            fix_after_image = test_inputs["map_image_tsr"].to(device)
            save_image( fix_pre_image, os.path.join(args.results_dir, args.exper_name) + "/fix_pre_image.png" )
            save_image( fix_after_image, os.path.join(args.results_dir, args.exper_name) + "/fix_after_image.png" )
            break

        with torch.no_grad():
            #G_z = model_G( fix_pre_image )
            G_z = model_G( fix_after_image )

        save_image( tensor = G_z[0], filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}_batch0.png".format( epoch ) )
        save_image( tensor = G_z, filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}_batchAll.png".format( epoch ) )

        fake_images_historys.append(G_z[0].transpose(0,1).transpose(1,2).cpu().clone().numpy())
        save_image_historys_gif( fake_images_historys, os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}.gif".format( epoch ) )        

    save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "G", 'G_final.pth'), iterations )
    save_checkpoint( model_D, device, os.path.join(args.save_checkpoints_dir, args.exper_name, "D", 'D_final.pth'), iterations )
    print("Finished Training Loop.")
