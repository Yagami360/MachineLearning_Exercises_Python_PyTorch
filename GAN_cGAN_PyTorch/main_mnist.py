# -*- coding:utf-8 -*-

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#import pickle
import scipy.misc

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader

import torchvision      # 画像処理関連
import torchvision.transforms as transforms
from torchvision.utils import save_image

# 自作モジュール
from ConditionalDCGANforMNIST import ConditionalDCGANforMNIST

#--------------------------------
# 設定可能な定数
#--------------------------------
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET_PATH = "./dataset"    # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数
BATCH_SIZE = 128              # ミニバッチサイズ
NUM_INPUT_NOIZE_Z = 62        # 生成器に入力するノイズ z の次数


def main():
    """
    cGAN（DCGANベース）による画像の自動生成
    ・学習用データセットは、MNIST
    ・MNIST に最適化されたネットワークで学習
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
    print( "NUM_EPOCHES : ", NUM_EPOCHES )
    print( "LEARNING_RATE : ", LEARNING_RATE )
    print( "BATCH_SIZE : ", BATCH_SIZE )
    print( "NUM_FEATURE_MAPS : ", NUM_FEATURE_MAPS )
    print( "NUM_INPUT_NOIZE_Z : ", NUM_INPUT_NOIZE_Z )

    #===================================
    # 実行 Device の設定
    #===================================
    if( DEVICE == "GPU" ):
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
    import random
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    # データをロードした後に行う各種前処理の関数を構成を指定する。
    transform = transforms.Compose(
        [
            transforms.ToTensor(),   # Tensor に変換
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
        download = True,
    )

    ds_test = torchvision.datasets.MNIST(
        root = DATASET_PATH,
        train = False,
        transform = transform,
        target_transform = None,
        download = True
    )

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
    #print( "dloader_train :", dloader_train )
    #print( "dloader_test :", dloader_test )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model = ConditionalDCGANforMNIST(
            device = device,
            n_epoches = NUM_EPOCHES,
            learing_rate = LEARNING_RATE,
            batch_size = BATCH_SIZE,
            n_input_noize_z = NUM_INPUT_NOIZE_Z
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
    model.fit( dloader = dloader_train, n_sava_step = NUM_SAVE_STEP )

    #===================================
    # 学習結果の描写処理
    #===================================
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
    #plt.xlim( 0, len(model.loss_G_history) )
    #plt.ylim( [0, 1.05] )
    plt.xlabel( "iterations" )
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        "cGANforMNIST_Loss_epoches{}_lr{}_batchsize{}.png".format( NUM_EPOCHES, LEARNING_RATE, BATCH_SIZE ),  
        dpi = 300, bbox_inches = "tight"
    )
    plt.show()


    #-------------------------------------------------------------------
    # 学習済み GAN に対し、自動生成画像を表示
    #-------------------------------------------------------------------
    images = model.generate_images( n_samples = 64, b_transformed = False )
    #print( "images.size() : ", images.size() )    # (64, 1, 28, 28)

    save_image( 
        tensor = images, 
        filename = "cGANforMNIST_Image_epoches{}_lr{}_batchsize{}.png".format( NUM_EPOCHES, LEARNING_RATE, BATCH_SIZE )
    )

    """
    images = model.generate_images( n_samples = 64, b_transformed = True )
    scipy.misc.imsave( 
        "CGANforMNIST_Image_epoches{}_lr{}_batchsize{}.png".format( NUM_EPOCHES, LEARNING_RATE, BATCH_SIZE ),
        np.vstack(
            np.array( [ np.hstack(img) for img in images ] )
        )
    )
    """

    print("Finish main()")
    print( "終了時間：", datetime.now() )

    return


    
if __name__ == '__main__':
     main()