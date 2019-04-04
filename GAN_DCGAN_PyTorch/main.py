# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision      # 画像処理関連

# 自作モジュール
from DeepConvolutionalGAN import DeepConvolutionalGAN


#--------------------------------
# 設定可能な定数
#--------------------------------
#DEVICE = "CPU"               # 使用デバイス ("CPU" or "GPU")
DEVICE = "GPU"                # 使用デバイス ("CPU" or "GPU")

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
    #======================================================================

    #======================================================================
    # データの前処理
    #======================================================================

    #------------------------------------
    # DataLoader への変換
    #------------------------------------


    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    model = DeepConvolutionalGAN(
        device = device,
        learing_rate = LEARNING_RATE,
        batch_size = BATCH_SIZE
    )
    model.print( "after init()" )

    #print( "model.device() :", model.device )

    #---------------------------------------------------------------
    # 損失関数を設定
    #---------------------------------------------------------------


    #---------------------------------------------------------------
    # optimizer を設定
    #---------------------------------------------------------------
    #model.optimizer()

    #======================================================================
    # モデルの学習フェイズ
    #======================================================================

    #======================================================================
    # モデルの推論フェーズ
    #======================================================================


    print("Finish main()")
    print( "終了時間：", datetime.now() )

    return


    
if __name__ == '__main__':
     main()