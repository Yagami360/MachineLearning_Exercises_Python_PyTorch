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
from networks import UNet
from map2aerial_dataset import Map2AerialDataset, Map2AerialDataLoader
from utils import save_checkpoint, load_checkpoint
from utils import board_add_image, board_add_images
from utils import save_image_historys_gif

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="UNet_train", help="実験名")
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
    parser.add_argument('--lr', type=float, default=0.0002, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--image_size', type=int, default=64, help="入力画像のサイズ（pixel単位）")
    parser.add_argument('--n_fmaps', type=int, default=64, help="特徴マップの枚数")
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
    model = UNet( 
        n_in_channels = 3, n_out_channels = 3,
        n_fmaps = args.n_fmaps,
    ).to( device )
        
    if( args.debug ):
        print( "model :\n", model )

    # モデルを読み込む
    if not args.load_checkpoints_dir == '' and os.path.exists(args.load_checkpoints_dir):
        init_step = load_checkpoint(model, device, args.load_checkpoints_dir, "model_final.pth" )

    #======================================================================
    # optimizer の設定
    #======================================================================
    optimizer = optim.Adam(
        params = model.parameters(),
        lr = args.lr, betas = (args.beta1,args.beta2)
    )
    
    #======================================================================
    # loss 関数の設定
    #======================================================================
    loss_fn = nn.MSELoss()    # when not use sigmoid in Discriminator

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
            model.train()

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
            # モデル の fitting 処理
            #====================================================
            #----------------------------------------------------
            # 学習用データをモデルに流し込む
            # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
            #----------------------------------------------------
            output = model( pre_image )
            if( args.debug and n_print > 0 ):
                print( "output.shape :", output.shape )

            #----------------------------------------------------
            # 損失関数を計算する
            #----------------------------------------------------
            loss = loss_fn( output, after_image )

            #----------------------------------------------------
            # ネットワークの更新処理
            #----------------------------------------------------
            # 勾配を 0 に初期化（この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
            optimizer.zero_grad()

            # 勾配計算
            loss.backward()

            # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
            optimizer.step()

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_display_step == 0 ) ):
                board_train.add_scalar('Model/loss', loss.item(), iterations)
                print( "epoch={}, iters={}, loss={:.5f}".format(epoch, iterations, loss) )

                visuals = [
                    [pre_image, after_image, output],
                ]
                board_add_images(board_train, 'images', visuals, iterations)

            #====================================================
            # test loss の表示
            #====================================================
            if( step == 0 or ( step % args.n_display_test_step == 0 ) ):
                model.eval()

                n_test_loop = 0
                test_iterations = 0
                loss_total = 0
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
                        output = model( pre_image )

                    #----------------------------------------------------
                    # 損失関数を計算する
                    #----------------------------------------------------
                    loss = loss_fn( output, after_image )

                    # total
                    loss_total += loss.item()

                    if( test_iterations > args.n_test ):
                        break

                board_test.add_scalar('Model/loss', (loss_total/n_test_loop), iterations)

                visuals = [
                    [pre_image, after_image, output],
                ]
                """
                if( args.debug and n_print > 0 ):
                    for col, vis_item_row in enumerate(visuals):
                        for row, vis_item in enumerate(vis_item_row):
                            print("[test] vis_item[{}][{}].shape={} :".format(row,col,vis_item.shape) )
                """
                board_add_images(board_test, "images_test", visuals, iterations)

            #====================================================
            # モデルの保存
            #====================================================
            if( ( step % args.n_save_step == 0 ) ):
                save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth'), iterations )
                print( "saved checkpoints" )

            n_print -= 1
        
        #====================================================
        # 各 Epoch 終了後の処理
        #====================================================
        # 出力画像の生成＆保存
        model.eval()
        for test_inputs in dloader_test :
            fix_pre_image = test_inputs["aerial_image_tsr"].to(device)
            fix_after_image = test_inputs["map_image_tsr"].to(device)
            save_image( fix_pre_image, os.path.join(args.results_dir, args.exper_name) + "/fix_pre_image.png" )
            save_image( fix_after_image, os.path.join(args.results_dir, args.exper_name) + "/fix_after_image.png" )
            break

        with torch.no_grad():
            output = model( fix_pre_image )

        save_image( tensor = output[0], filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}_batch0.png".format( epoch ) )
        save_image( tensor = output, filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}_batchAll.png".format( epoch ) )

        fake_images_historys.append(output[0].transpose(0,1).transpose(1,2).cpu().clone().numpy())
        save_image_historys_gif( fake_images_historys, os.path.join(args.results_dir, args.exper_name) + "/fake_image_epoches{}.gif".format( epoch ) )        

    save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth'), iterations )
    print("Finished Training Loop.")
