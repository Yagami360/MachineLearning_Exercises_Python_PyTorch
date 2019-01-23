# -*- coding:utf-8 -*-
# Anaconda 5.0.1 環境
# PyTorch : 1.0.0
# scikit-learn : 0.20.2

import numpy as np
import matplotlib.pyplot as plt

# scikit-learn
import sklearn
from sklearn import datasets                            # 
from sklearn.model_selection import train_test_split    # scikit-learn の train_test_split関数の new-version


# PyTorch
import torch
from torch  import nn



# 設定可能な定数
BATCH_SIZE = 32         # バッチサイズ
NUM_EPOCHES = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10

def main():
    """
	Sequental を用いたニューラルネットワークの構成（Keras風の書き方）
    cf : (KerasでのSequental) https://github.com/Yagami360/MachineLearning_Exercises_Python_Keras/blob/master/Processing_Keras/main1.py
    """
    print("Start main()")

    # ライブラリのバージョン確認
    print( "PyTorch :", torch.__version__ )
    print( "sklearn :", sklearn.__version__ )

    #======================================================================
    # アルゴリズム（モデル）のパラメータを設定
    # Set algorithm parameters.
    # ex) learning_rate = 0.01  iterations = 1000
    #======================================================================

    #======================================================================
    # データセットを読み込み or 生成
    # Import or generate data.
    #======================================================================
    #======================================================================
    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    #======================================================================

    #======================================================================
    # データを変換、正規化
    # Transform and normalize data.
    # ex) data = tf.nn.batch_norm_with_global_normalization(...)
    #======================================================================

    #======================================================================
    # モデルの構造を定義する。
    # Define the model structure.
    #======================================================================

    #---------------------------------------------------------------
    # keras.layers.Sequential.add() メソッドで簡単にレイヤーを追加
    # この例では、パーセプトロン
    #---------------------------------------------------------------

    #---------------------------------------------------------------
    # optimizer, loss を設定
    #---------------------------------------------------------------

    #======================================================================
    # モデルの初期化と学習（トレーニング）
    #======================================================================

    #======================================================================
    # モデルの評価
    # (Optional) Evaluate the model.
    #======================================================================


    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()