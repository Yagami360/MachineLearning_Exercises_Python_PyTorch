# -*- coding:utf-8 -*-
import os
import argparse
from datetime import datetime
import numpy as np
from PIL import Image
#import pickle
import scipy.misc
import random

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader

import torchvision      # 画像処理関連
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 自作モジュール
from model import WassersteinGAN


if __name__ == '__main__':
    """
    WGAN による画像の自動生成
    ・学習用データセットは、MNIST / CIFAR-10
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="WGAN_train", help="実験名")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="データセットのディレクトリ")
    parser.add_argument('--result_dir', type=str, default="result", help="結果を保存するディレクトリ")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--dataset', choices=['mnist', 'cifar-10'], default="mnist", help="データセットの種類（MNIST or CIFAR-10）")
    parser.add_argument('--n_epoches', type=int, default=10, help="エポック数")
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=0.00005, help="学習率")
    parser.add_argument('--image_size', type=int, default=64, help="入力画像のサイズ（pixel単位）")
    parser.add_argument('--n_channels', type=int, default=1, help="入力画像のチャンネル数")
    parser.add_argument('--n_fmaps', type=int, default=64, help="特徴マップの枚数")
    parser.add_argument('--n_input_noize_z', type=int, default=100, help="生成器に入力するノイズ z の次数")
    parser.add_argument('--n_critic', type=int, default=5, help="クリティックの更新回数")
    parser.add_argument('--w_clamp_upper', type=float, default=0.01, help="重みクリッピングの下限値")
    parser.add_argument('--w_clamp_lower', type=float, default=-0.01, help="重みクリッピングの下限値")
    parser.add_argument('--n_save_step', type=int, default=100, help="生成画像の保存間隔（イテレーション単位）")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    #===================================
    # 実行条件の出力
    #===================================
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "----------------------------------------------" )
    print( "開始時間：", datetime.now() )
    print( "PyTorch version :", torch.__version__ )
    for key, value in vars(args).items():
        print('%s: %s' % (str(key), str(value)))
    print('-------------- End ----------------------------')

    if not( os.path.exists(args.result_dir) ):
        os.mkdir(args.result_dir)
    if not( os.path.exists(args.tensorboard_dir) ):
        os.mkdir(args.tensorboard_dir)

    #===================================
    # 実行 Device の設定
    #===================================
    if( args.device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            print( "実行デバイス :", device)
            print( "GPU名 :", torch.cuda.get_device_name(0))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device)
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device)

    print( "----------------------------------------------" )

    # seed 値の固定
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    # データをロードした後に行う各種前処理の関数を構成を指定する。
    if( args.dataset == "mnist" ):
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=Image.LANCZOS ),
                transforms.ToTensor(),   # Tensor に変換]
                transforms.Normalize((0.5,), (0.5,)),   # 1 channel 分
            ]
        )

    elif( args.dataset == "cifar-10" ):
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=Image.LANCZOS ),
                transforms.ToTensor(),   # Tensor に変換
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    else:
        print( "Warning: Invalid dataset" )

    #---------------------------------------------------------------
    # data と label をセットにした TensorDataSet の作成
    #---------------------------------------------------------------
    if( args.dataset == "mnist" ):
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
        print( "WARNING: Inavlid dataset" )

    #print( "ds_train :", ds_train ) # MNIST : torch.Size([60000, 28, 28]) , CIFAR-10 : (50000, 32, 32, 3)
    #print( "ds_test :", ds_test )

    #---------------------------------------------------------------
    # TensorDataset → DataLoader への変換
    # DataLoader に変換することで、バッチ処理が行える。
    # DataLoader クラスは、dataset と sampler クラスを持つ。
    # sampler クラスには、ランダムサンプリングや重点サンプリングなどがある
    #---------------------------------------------------------------
    dloader_train = DataLoader(
        dataset = ds_train,
        batch_size = args.batch_size,
        shuffle = True
    )

    dloader_test = DataLoader(
        dataset = ds_test,
        batch_size = args.batch_size,
        shuffle = False
    )
    
    # [MNIST]
    # Number of datapoints: 60000
    # dloader_train.datset
    # dloader_train.sampler = <RandomSampler, len() = 60000>
    #print( "dloader_train :", dloader_train )
    #print( "dloader_test :", dloader_test )
    
    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model = WassersteinGAN(
        device = device,
        exper_name = args.exper_name,
        n_epoches = args.n_epoches,
        learing_rate = args.lr,
        batch_size = args.batch_size,
        n_channels = args.n_channels,
        n_fmaps = args.n_fmaps,
        n_input_noize_z = args.n_input_noize_z,
        n_critic = args.n_critic,
        w_clamp_lower = args.w_clamp_lower,
        w_clamp_upper = args.w_clamp_upper
    )

    model.print( "after init()" )

    #---------------------------------------------------------------
    # 損失関数を設定
    #---------------------------------------------------------------
    #model.loss()

    #---------------------------------------------------------------
    # optimizer を設定
    #---------------------------------------------------------------
    #model.optimizer()

    #======================================================================
    # モデルの学習フェイズ
    #======================================================================
    board = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
    model.fit( dloader = dloader_train, n_sava_step = args.n_save_step, result_path = args.result_dir, board = board )

    #===================================
    # 学習結果の描写処理
    #===================================
    #-----------------------------------
    # 損失関数の plot
    #-----------------------------------
    plt.clf()
    plt.plot(
        range( 0, len(model.loss_G_history) ), model.loss_G_history,
        label = "loss : Generator",
        linestyle = '-',
        linewidth = 0.2,
        color = 'red'
    )
    plt.plot(
        range( 0, len(model.loss_C_history) ), model.loss_C_history,
        label = "loss : Critic",
        linestyle = '-',
        linewidth = 0.2,
        color = 'blue'
    )
    plt.title( "loss" )
    plt.legend( loc = 'best' )
    #plt.xlim( 0, len(model.loss_G_history) )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "iterations" )
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.result_dir, args.exper_name + "_loss_epoches{}_lr{}_batchsize{}.png".format( args.n_epoches, args.lr, args.batch_size ) ),  
        dpi = 300, bbox_inches = "tight"
    )
    plt.show()

    #-------------------------------------------------------------------
    # 学習済み DCGAN に対し、自動生成画像を表示
    #-------------------------------------------------------------------
    images = model.generate_images( n_samples = 64, b_transformed = False )
    #print( "images.size() : ", images.size() )    # (64, 1, 28, 28)

    save_image( 
        tensor = images, 
        filename = os.path.join(args.result_dir, args.exper_name + "_image_epoches{}_lr{}_batchsize{}.png".format( args.n_epoches, args.lr, args.batch_size ) )
    )

    print("Finish main()")
    print( "終了時間：", datetime.now() )
