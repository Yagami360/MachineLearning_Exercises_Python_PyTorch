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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="UNet_test", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset_dir', type=str, default="dataset/maps", help="データセットのディレクトリ")
    parser.add_argument('--results_dir', type=str, default="results", help="生成画像の出力ディレクトリ")
    parser.add_argument('--load_checkpoints_dir', type=str, default="", help="モデルの読み込みディレクトリ")
    parser.add_argument('--n_samplings', type=int, default=100, help="サンプリング数")
    parser.add_argument('--batch_size', type=int, default=32, help="バッチサイズ")
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

    # seed 値の固定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    ds_test = Map2AerialDataset( args.dataset_dir, "val", args.image_size, args.image_size, args.debug )
    dloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size, shuffle=False )

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
        init_step = load_checkpoint(model, device, os.path.join(args.load_checkpoints_dir, "model_final.pth") )
    
    #======================================================================
    # モデルの推論処理
    #======================================================================
    print("Starting Test Loop...")
    n_print = 1
    model.eval()
    # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
    for step, inputs in enumerate( tqdm( dloader_test, desc = "Samplings" ) ):
        # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する
        # （後の計算で、shape の不一致をおこすため）
        if inputs["aerial_image_tsr"].shape[0] != args.batch_size:
            break

        # ミニバッチデータを GPU へ転送
        pre_image = inputs["aerial_image_tsr"].to(device)
        after_image = inputs["map_image_tsr"].to(device)
        #save_image( pre_image, "pre_image.png" )
        #save_image( after_image, "after_image.png" )

        #====================================================
        # 生成器 G の 推論処理
        #====================================================
        with torch.no_grad():
            output = model( pre_image )
            if( args.debug and n_print > 0 ):
                print( "output.shape :", output.shape )

        n_print -= 1
        
        #---------------------
        # 出力画像の生成＆保存
        #---------------------
        save_image( tensor = output, filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_step{}_batchAll.png".format( step ) )
        n_print -= 1
        if( step >= args.n_samplings ):
            break

    print("Finished Test Loop.")