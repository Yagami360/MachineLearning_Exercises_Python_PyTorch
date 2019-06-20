# -*- coding:utf-8 -*-
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os


# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader

import torchvision      # 画像処理関連
import torchvision.transforms as transforms
from torchvision.utils import save_image
import tensorboardX as tbx

# 自作モジュール
from ProgressiveGANforMNIST import ProgressiveGANforMNIST
from ProgressiveGANforCIFAR10 import ProgressiveGANforCIFAR10


if __name__ == '__main__':
    """
    Progressive GAN による高解像度画像の自動生成
    """
    print("Start main()")

    # コマンドライン引数の取得
    parser = argparse.ArgumentParser()
    parser.add_argument( "--device", type = str, default = "GPU" )              # GPU | CPU
    parser.add_argument( "--run_mode", type = str, default = "train" )          # train | add_train | test
    parser.add_argument( "--dataset", type = str, default = "MNIST" )           # MNIST | CIFAR-10
    #parser.add_argument( "--dataset", type = str, default = "CIFAR-10" )        # 
    parser.add_argument( "--dataset_path", type = str, default = "./dataset" )  # 
    parser.add_argument( "--n_input_noize_z", type = int, default = 128 )
    parser.add_argument( "--init_image_size", type = int, default = 4 )
    parser.add_argument( "--final_image_size", type = int, default = 32 )
    #parser.add_argument( "--final_image_size", type = int, default = 64 )
    parser.add_argument( "--n_epoches", type = int, default = 10 )    
    parser.add_argument( "--batch_size", type = int, default = 16 )
    parser.add_argument( "--learning_rate", type = float, default = 0.001 )
    parser.add_argument( "--result_path", type = str, default = "./result" )
    args = parser.parse_args()
    
    # 実行条件の出力
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "----------------------------------------------" )
    print( "--device : ", args.device )
    print( "--run_mode : ", args.run_mode )
    print( "--dataset : ", args.dataset )
    print( "--dataset_path : ", args.dataset_path )
    print( "--n_input_noize_z : ", args.n_input_noize_z )
    print( "--init_image_size : ", args.init_image_size )
    print( "--final_image_size : ", args.final_image_size )
    print( "--n_epoches : ", args.n_epoches )
    print( "--batch_size : ", args.batch_size )
    print( "--learning_rate : ", args.learning_rate )
    print( "--result_path : ", args.result_path )

    #===================================
    # 実行 Device の設定
    #===================================
    if( args.device == "GPU" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            print( "実行デバイス :", device )
            print( "GPU名 :", torch.cuda.get_device_name(0) )
            print("torch.cuda.current_device() =", torch.cuda.current_device() )
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device )
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device )

    print( "----------------------------------------------" )

    # seed 値の固定
    import random
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    if( args.dataset == "MNIST" ):
        transform = transforms.Compose(
            [
                transforms.Resize( args.final_image_size ),
                transforms.ToTensor(),   # Tensor に変換
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
    elif( args.dataset == "CIFAR-10" ):
        transform = transforms.Compose(
            [
                transforms.Resize( args.final_image_size ),
                transforms.ToTensor(),   # Tensor に変換
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    else:
        print( "Warning: Invalid dataset" )

    #---------------------------------------------------------------
    # data と label をセットにした TensorDataSet の作成
    #---------------------------------------------------------------
    if( args.dataset == "MNIST" ):
        ds_train = torchvision.datasets.MNIST(
            root = args.dataset_path,
            train = True,
            transform = transform,      # transforms.Compose(...) で作った前処理の一連の流れ
            target_transform = None,    
            download = True,
        )

        ds_test = torchvision.datasets.MNIST(
            root = args.dataset_path,
            train = False,
            transform = transform,
            target_transform = None,
            download = True
        )
    elif( args.dataset == "CIFAR-10" ):
        ds_train = torchvision.datasets.CIFAR10(
            root = args.dataset_path,
            train = True,
            transform = transform,      # transforms.Compose(...) で作った前処理の一連の流れ
            target_transform = None,    
            download = True
        )

        ds_test = torchvision.datasets.CIFAR10(
            root = args.dataset_path,
            train = False,
            transform = transform,
            target_transform = None,
            download = True
        )
    else:
        print( "WARNING: Inavlid dataset" )

    #print( "ds_train :", ds_train )
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

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    if( args.dataset == "MNIST" ):
        model = ProgressiveGANforMNIST(
            device = device,
            n_epoches = args.n_epoches,
            batch_size = args.batch_size,
            learing_rate = args.learning_rate,
            n_input_noize_z = args.n_input_noize_z,
            init_image_size = args.init_image_size,
            final_image_size = args.final_image_size,
        )
        model.print( "after init()" )

    elif( args.dataset == "CIFAR-10" ):
        model = ProgressiveGANforCIFAR10(
            device = device,
            n_epoches = args.n_epoches,
            batch_size = args.batch_size,
            learing_rate = args.learning_rate,
            n_input_noize_z = args.n_input_noize_z,
            init_image_size = args.init_image_size,
            final_image_size = args.final_image_size,
        )
        model.print( "after init()" )

    else:
        print( "WARNING: Inavlid dataset" )

    #======================================================================
    # モデルの学習フェイズ
    #======================================================================
    if( args.run_mode == "train" ):
        model.fit( dloader = dloader_train, n_sava_step = 1000, result_path = args.result_path )
        #model.fit( dloader = dloader_test, n_sava_step = 1000, result_path = args.result_path )
    elif( args.run_mode == "add_train" ):
        model.load_model()
        model.fit( dloader = dloader_train, n_sava_step = 1000, result_path = args.result_path )
    elif( args.run_mode == "test" ):
        model.load_model()

    #===================================
    # 学習結果の描写処理
    #===================================
    if( args.run_mode != "test" ):
        #import sys
        #sys.exit(0)

        #-----------------------------------
        # 損失関数の plot
        #-----------------------------------
        plt.clf()
        plt.plot(
            range( 0, len(model.loss_G_history) ), model.loss_G_history,
            label = "loss_G",
            linestyle = '-',
            linewidth = 0.2,
            color = 'red'
        )
        plt.plot(
            range( 0, len(model.loss_D_history) ), model.loss_D_history,
            label = "loss_D",
            linestyle = '-',
            linewidth = 0.2,
            color = 'blue'
        )
        plt.title( "loss" )
        plt.legend( loc = 'best' )
        #plt.xlim( 0, len(model.loss_history) )
        #plt.ylim( [0, 1.05] )
        plt.xlabel( "iterations" )
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                args.result_path,     
                "PGGAN_Loss_epoches{}_lr{}_batchsize{}.png".format( args.n_epoches, args.learning_rate, args.batch_size )
            ),
            dpi = 300, bbox_inches = "tight"
        )
        plt.show()

    print("Finish main()")
    print( "終了時間：", datetime.now() )
