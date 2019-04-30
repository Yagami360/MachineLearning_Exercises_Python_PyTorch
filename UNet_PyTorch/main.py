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
from UNet import SemanticSegmentationwithUNet

#--------------------------------
# 設定可能な定数
#--------------------------------
DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
#DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")
DATASET_PATH = "./maps"       # 学習用データセットへのパス
NUM_SAVE_STEP = 1             # 自動生成画像の保存間隔（エポック単位）

NUM_EPOCHES = 10              # エポック数（学習回数）
LEARNING_RATE = 0.0002        # 学習率 (Default:0.0002)
BATCH_SIZE = 1                # ミニバッチサイズ
IMAGE_SIZE = 256              # 入力画像のサイズ（pixel単位）
NUM_CHANNELS = 1              # 入力画像のチャンネル数
NUM_FEATURE_MAPS = 64         # 特徴マップの枚数


def main():
    """
    U-Net による画像のセグメンテーション
    ・学習用データセットは、
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
    print( "IMAGE_SIZE : ", IMAGE_SIZE )
    print( "NUM_FEATURE_MAPS : ", NUM_FEATURE_MAPS )

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
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop( size = (IMAGE_SIZE, IMAGE_SIZE*2) ),
            transforms.ToTensor(),   # Tensor に変換]
            #transforms.Normalize((0.5,), (0.5,)),   # 1 channel 分
        ]
    )

    #---------------------------------------------------------------
    # data と label をセットにした TensorDataSet の作成
    #---------------------------------------------------------------
    dataset = torchvision.datasets.ImageFolder(
        root = DATASET_PATH,
        transform = transform,
        target_transform = None
    )
    print( "dataset :", dataset )

    #---------------------------------------------------------------
    # TensorDataset → DataLoader への変換
    # DataLoader に変換することで、バッチ処理が行える。
    # DataLoader クラスは、dataset と sampler クラスを持つ。
    # sampler クラスには、ランダムサンプリングや重点サンプリングなどがある
    #---------------------------------------------------------------
    dloader = DataLoader(
        dataset = dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )
    print( "dloader :", dloader )
    
    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model = SemanticSegmentationwithUNet(
        device = device,
        n_epoches = NUM_EPOCHES,
        learing_rate = LEARNING_RATE,
        batch_size = BATCH_SIZE,
        n_fmaps = NUM_FEATURE_MAPS
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
    model.fit( dloader = dloader, n_sava_step = NUM_SAVE_STEP )

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
        "UNet_Loss_epoches{}_lr{}_batchsize{}.png".format( NUM_EPOCHES, LEARNING_RATE, BATCH_SIZE ),  
        dpi = 300, bbox_inches = "tight"
    )
    plt.show()

    print("Finish main()")
    print( "終了時間：", datetime.now() )

    return


    
if __name__ == '__main__':
     main()