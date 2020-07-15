# -*- coding:utf-8 -*-
# Anaconda 4.3.0 環境

import os
import sys

import struct
import numpy

# Data Frame & IO 関連
import pandas
from io import StringIO

# scikit-learn ライブラリ関連
from sklearn import datasets                            # scikit-learn ライブラリのデータセット群
from sklearn.datasets import make_moons                 # 半月状のデータセット生成
from sklearn.datasets import make_circles               # 同心円状のデータセット生成

#from sklearn.cross_validation import train_test_split  # scikit-learn の train_test_split関数の old-version
from sklearn.model_selection import train_test_split    # scikit-learn の train_test_split関数の new-version
from sklearn.metrics import accuracy_score              # 正解率、誤識別率の計算用に使用

from sklearn.preprocessing import Imputer               # データ（欠損値）の保管用に使用
from sklearn.preprocessing import LabelEncoder          # 
from sklearn.preprocessing import OneHotEncoder         # One-hot encoding 用に使用
from sklearn.preprocessing import MinMaxScaler          # scikit-learn の preprocessing モジュールの MinMaxScaler クラス
from sklearn.preprocessing import StandardScaler        # scikit-learn の preprocessing モジュールの StandardScaler クラス


class MLPreProcess( object ):
    """
    機械学習用のデータの前処理を行うクラス
    
    [public] public アクセス可能なインスタスンス変数には, 便宜上変数名の最後にアンダースコア _ を付ける.
    
    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    #---------------------------------------------------------
    # 検証用サンプルデータセットを出力する関数群
    #---------------------------------------------------------

    @staticmethod
    def load_mnist( path, kind = "train" ):
        """
        検証データ用の MNIST データを読み込む。
        [Input]
            path : str
                MNIST データセットが格納されているフォルダへのパス
            kind : str
                読み込みたいデータの種類（トレーニング用データ or テスト用データ）
                "train" : トレーニング用データ
                "t10k" : テスト用データ
        [Output]
            images : [n_samples = 60,000, n_features = 28*28 = 784]
                トレーニングデータ用の画像データ
            labels : [n_samples = 60,000,]
                トレーニングデータ用のラベルデータ（教師データ）
                0~9 の数字ラベル
        """
        # path の文字列を結合して、MIST データへのパスを作成
        # (kind = train) %s → train-images.idx3-ubyte, train-labels.idx1-ubyte
        # (kind = t10k)  %s → t10k-images.idx3-ubyte,  t10k-labels.idx1-ubyte
        labels_path = os.path.join( path, "%s-labels.idx1-ubyte" % kind )
        images_path = os.path.join( path, "%s-images.idx3-ubyte" % kind )

        #------------------------------------------
        # open() 関数と with 構文でラベルデータ（教師データ）の読み込み
        # "rb" : バイナリーモードで読み込み
        #------------------------------------------
        with open( labels_path, 'rb' ) as lbpath:
            # struct.unpack(...) : バイナリーデータを読み込み文字列に変換
            # magic : マジックナンバー （先頭 から 4byte）
            # num : サンプル数（magicの後の 4byte）
            magic, n = \
            struct.unpack(
                '>II',           # > : ビッグエンディアン, I : 符号なし整数, >II : 4byte + 4byte
                lbpath.read(8)   # 8byte
            )
            
            # numpy.fromfile(...) : numpy 配列にバイトデータを読み込んむ
            # dtype : numpy 配列のデータ形式
            labels = numpy.fromfile( file = lbpath, dtype = numpy.uint8 )

        #------------------------------------------
        # open() 関数と with 構文で画像データの読み込む
        # "rb" : バイナリーモードで読み込み
        #------------------------------------------
        with open( images_path, "rb" ) as imgpath:
            # struct.unpack(...) : バイナリーデータを読み込み文字列に変換
            # magic : マジックナンバー （先頭 から 4byte）
            # num : サンプル数（magicの後の 4byte）
            # rows : ?
            # cols : ?
            magic, num, rows, cols = \
            struct.unpack(
                ">IIII",           # > : ビッグエンディアン, I : 符号なし整数, >IIII : 4byte + 4byte + 4byte + 4byte
                imgpath.read(16)   # 16byte
            )

            # numpy.fromfile(...) : numpy 配列にバイトデータを読み込んでいき,
            # 読み込んだデータを shape = [labels, 784] に reshape
            images = numpy.fromfile( file = imgpath, dtype = numpy.uint8 )
            images = images.reshape( len(labels), 784 )
       
            
        return images, labels
