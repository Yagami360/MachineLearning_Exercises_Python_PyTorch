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

# 自作モジュール
from ResNet import ResNetClassifier


if __name__ == '__main__':
    """
    ResNet による画像分類タスク
    """
    print("Start main()")

    # コマンドライン引数の取得
    parser = argparse.ArgumentParser()
    parser.add_argument( "--device", type = str, default = "GPU" )
    parser.add_argument( "--dataset", type = str, default = "CIFAR-10" )
    parser.add_argument( "--dataset_path", type = str, default = "./dataset" )
    parser.add_argument( "--image_size", type = int, default = 224 )
    parser.add_argument( "--n_classes", type = int, default = 10 )
    parser.add_argument( "--n_epoches", type = int, default = 50 )
    parser.add_argument( "--batch_size", type = int, default = 32 )
    parser.add_argument( "--learning_rate", type = float, default = 0.001 )
    parser.add_argument( "--result_path", type = str, default = "./result" )
    args = parser.parse_args()
    
    # 実行条件の出力
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "----------------------------------------------" )
    print( "開始時間：", datetime.now() )
    print( "PyTorch version:", torch.__version__ )
    print( "--device : ", args.device )
    print( "--dataset : ", args.dataset )
    print( "--dataset_path : ", args.dataset_path )
    print( "--image_size : ", args.image_size )
    print( "--n_classes : ", args.n_classes )
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
                transforms.Resize( args.image_size ),
                transforms.ToTensor(),   # Tensor に変換
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        
    elif( args.dataset == "CIFAR-10" ):
        transform = transforms.Compose(
            [
                transforms.Resize( args.image_size ),
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
    model = ResNetClassifier(
        device = device,
        n_classes = args.n_classes,
        n_epoches = args.n_epoches,
        batch_size = args.batch_size,
        learing_rate = args.learning_rate,
    )

    model.print( "after init()" )

    #======================================================================
    # モデルの学習フェイズ
    #======================================================================
    model.fit( dloader = dloader_train, n_sava_step = 1, result_path = args.result_path )

    #===================================
    # 学習結果の描写処理
    #===================================
    #-----------------------------------
    # 損失関数の plot
    #-----------------------------------
    plt.clf()
    plt.plot(
        range( 0, len(model.loss_history) ), model.loss_history,
        linestyle = '-',
        linewidth = 0.2,
        color = 'red'
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
            "ResNet18_Loss_epoches{}_lr{}_batchsize{}.png".format( args.n_epoches, args.learning_rate, args.batch_size )
        ),
        dpi = 300, bbox_inches = "tight"
    )
    plt.show()


    print("Finish main()")
    print( "終了時間：", datetime.now() )
