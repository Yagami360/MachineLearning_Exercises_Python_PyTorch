# -*- coding:utf-8 -*-
import argparse
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image
from math import ceil

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision      # 画像処理関連
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 自作クラス
from networks import ProgressiveGenerator, ProgressiveDiscriminator
from utils import save_checkpoint, load_checkpoint
from utils import board_add_image, board_add_images
from utils import save_image_historys_gif

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="PGGAN_train", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset', choices=['mnist', 'cifar-10'], default="mnist", help="データセットの種類（MNIST or CIFAR-10）")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="データセットのディレクトリ")
    parser.add_argument('--results_dir', type=str, default="results", help="生成画像の出力ディレクトリ")
    parser.add_argument('--load_checkpoints_dir', type=str, default="", help="モデルの読み込みディレクトリ")
    parser.add_argument('--n_samplings', type=int, default=100, help="サンプリング数")
    parser.add_argument('--batch_size', type=int, default=63, help="バッチサイズ")
    parser.add_argument("--init_image_size", type = int, default = 4 )
    parser.add_argument("--final_image_size", type = int, default = 32 )
    parser.add_argument('--n_input_noize_z', type=int, default=128, help="生成器に入力するノイズ z の次数")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード値")
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
    pass

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    # Genrator
    if( args.dataset == "mnist" ):
        model_G = ProgressiveGenerator(
            init_image_size = args.init_image_size,
            final_image_size = args.final_image_size,
            n_input_noize_z = args.n_input_noize_z,
            n_rgb = 1,
        ).to( device )
    else:
        model_G = ProgressiveGenerator(
            init_image_size = args.init_image_size,
            final_image_size = args.final_image_size,
            n_input_noize_z = args.n_input_noize_z,
            n_rgb = 3,
        ).to( device )

    # Discriminator
    if( args.dataset == "mnist" ):
        model_D = ProgressiveDiscriminator(
            init_image_size = args.init_image_size,
            final_image_size = args.final_image_size,
            n_fmaps = args.n_input_noize_z,
            n_rgb = 1,
        ).to( device )
    else:
        model_D = ProgressiveDiscriminator( 
            init_image_size = args.init_image_size,
            final_image_size = args.final_image_size,
            n_fmaps = args.n_input_noize_z,
            n_rgb = 3,
        ).to( device )
        
    if( args.debug ):
        print( "model_G :\n", model_G )
        print( "model_D :\n", model_D )

    # モデルを読み込む
    if not args.load_checkpoints_dir == '' and os.path.exists(args.load_checkpoints_dir):
        init_step = load_checkpoint(model_G, device, os.path.join(args.load_checkpoints_dir, "G", "G_final.pth") )
        init_step = load_checkpoint(model_D, device, os.path.join(args.load_checkpoints_dir, "D", "D_final.pth") )

    #======================================================================
    # モデルの学習処理
    #======================================================================
    # 入力ノイズ z
    input_noize_z = torch.randn( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )
    if( args.debug ):
        print( "input_noize_z.shape :", input_noize_z.shape )

    #
    final_progress = float(np.log2(args.final_image_size)) -2

    #======================================================================
    # サンプリング数分推論
    #======================================================================
    print("Starting Training Loop...")
    n_print = 1
    seed = args.seed
    for step in tqdm( range(args.n_samplings), desc = "Samplings" ):
        model_G.eval()
        model_D.eval()

        # seed 値の変更
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 入力ノイズ z の再生成
        input_noize_z = torch.randn( size = (args.batch_size, args.n_input_noize_z,1,1) ).to( device )

        #====================================================
        # 生成器 G の 推論処理
        #====================================================
        with torch.no_grad():
            # G(z) : 生成器から出力される偽物画像
            G_z = model_G( input_noize_z, progress=final_progress )

        #---------------------
        # 出力画像の生成＆保存
        #---------------------
        save_image( tensor = G_z, filename = os.path.join(args.results_dir, args.exper_name) + "/fake_image_seed{}_batchAll.png".format( seed ) )
        seed += 1
        n_print -= 1

    print("Finished Test Loop.")