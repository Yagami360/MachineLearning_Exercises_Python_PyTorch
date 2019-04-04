# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader

import torchvision      # 画像処理関連
import torchvision.transforms as transforms

# 自作モジュール
from DeepConvolutionalGAN import DeepConvolutionalGAN


#--------------------------------
# 設定可能な定数
#--------------------------------
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")

DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_EPOCHES = 25              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
BATCH_SIZE = 32               # ミニバッチサイズ (Default:32)


def main():
    """
    DCGAN による画像の自動生成
    ・学習用データセットは、MNIST
    """
    print("Start main()")
    
    # バージョン確認
    print( "PyTorch :", torch.__version__ )

    # 実行条件の出力
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "----------------------------------------------" )
    print( "開始時間：", datetime.now() )
    print( "DEVICE : ", DEVICE )
    print( "LEARNING_RATE : ", LEARNING_RATE )
    print( "BATCH_SIZE : ", BATCH_SIZE )

    #===================================
    # 実行 Device の設定
    #===================================
    if( DEVICE == "GPU" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
    else:
        device = torch.device( "cpu" )

    print( "実行デバイス :", device)
    print( "GPU名 :", torch.cuda.get_device_name(0))
    print( "----------------------------------------------" )

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    # データをロードした後に行う各種前処理の関数を構成を指定する。
    transform = transforms.Compose(
        [
            transforms.ToTensor()   # Tensor に変換
        ]
    )
    
    #---------------------------------------------------------------
    # data と label をセットにした TensorDataSet の作成
    #---------------------------------------------------------------
    ds_train = torchvision.datasets.MNIST(
        root = DATASET_PATH,
        train = True,
        transform = transform,      # transforms.Compose(...) で作った前処理の一連の流れ
        target_transform = None,    
        download = True
    )

    ds_test = torchvision.datasets.MNIST(
        root = DATASET_PATH,
        train = False,
        transform = transform,
        target_transform = None,
        download = True
    )

    print( "ds_train :", ds_train )
    print( "ds_test :", ds_test )

    #---------------------------------------------------------------
    # TensorDataset → DataLoader への変換
    # DataLoader に変換することで、バッチ処理が行える。
    # DataLoader クラスは、dataset と sampler クラスを持つ。
    # sampler クラスには、ランダムサンプリングや重点サンプリングなどがある
    #---------------------------------------------------------------
    dloader_train = DataLoader(
        dataset = ds_train,
        batch_size = BATCH_SIZE,
        shuffle = True
    )

    dloader_test = DataLoader(
        dataset = ds_test,
        batch_size = BATCH_SIZE,
        shuffle = False
    )
    
    # Number of datapoints: 60000
    # dloader_train.datset
    # dloader_train.sampler = <RandomSampler, len() = 60000>
    # dloader_train[0] = [<Tensor, len() = 32>, <Tensor, len() = 32>]
    # dloader_train[1874] = [<Tensor, len() = 32>, <Tensor, len() = 32>]
    print( "dloader_train :", dloader_train )
    print( "dloader_test :", dloader_test )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model = DeepConvolutionalGAN(
        device = device,
        n_epoches = NUM_EPOCHES,
        learing_rate = LEARNING_RATE,
        batch_size = BATCH_SIZE
    )
    model.print( "after init()" )

    #print( "model.device() :", model.device )

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
    model.fit( data_loader = dloader_train )

    #======================================================================
    # モデルの推論フェーズ
    #======================================================================


    print("Finish main()")
    print( "終了時間：", datetime.now() )

    return


    
if __name__ == '__main__':
     main()