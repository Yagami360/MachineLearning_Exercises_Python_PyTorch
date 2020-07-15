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
from torch.utils.data import TensorDataset, DataLoader
from torch  import nn   # ネットワークの構成関連
import torchvision      # 画像処理関連

# 自作クラス
from MLPreProcess import MLPreProcess

#======================================================================
# アルゴリズム（モデル）のパラメータを設定
#======================================================================
# 設定可能な定数
BATCH_SIZE = 64         # バッチサイズ
NUM_EPOCHES = 2        #
LEARNING_RATE = 0.0005   #
NUM_CLASSES = 10        # MNIST の 0~9 の数字


def main():
    """
	Sequental を用いたニューラルネットワークの構成（Keras風の書き方）
    cf : (tensorFlow) https://github.com/Yagami360/MachineLearning_Exercises_Python_TensorFlow/blob/master/ProcessingForMachineLearning_TensorFlow/main5.py
    cf : (KerasでのSequental) https://github.com/Yagami360/MachineLearning_Exercises_Python_Keras/blob/master/Processing_Keras/main1.py
    """
    print("Start main()")

    # ライブラリのバージョン確認
    print( "PyTorch :", torch.__version__ )
    print( "sklearn :", sklearn.__version__ )
    
    #======================================================================
    # データセットを読み込み or 生成
    #======================================================================
    # MNIST データが格納されているフォルダへのパス
    mnist_path = "C:\Data\MachineLearning_DataSet\MNIST"

    # データセットをトレーニングデータ、テストデータ、検証データセットに分割
    X_train, y_train = MLPreProcess.load_mnist( mnist_path, "train" )
    X_test, y_test = MLPreProcess.load_mnist( mnist_path, "t10k" )


    #mnist = sklearn.datasets.fetch_mldata( "MNIST original", data_home = mnist_path )
    #X_features = mnist.data
    #y_labels = mnist.target

    """
    # Dataset
    ds_train = torchvision.datasets.MNIST(
        root = mnist_path,
        train = True,
        transform = None,
        target_transform = None,
        download = True
    )

    ds_test = torchvision.datasets.MNIST(
        root = mnist_path,
        train = False,
        transform = None,
        target_transform = None,
        download = True
    )
    """

    #======================================================================
    # データの前処理
    #======================================================================
    # 0 ~ 255 → 0.0f ~ 1.0f
    X_train = X_train / 255
    X_test = X_test / 255

    print( "X_train <numpy> :\n", X_train )
    print( "y_train <numpy> :\n", y_train )

    #-----------------------------------
    # 取得したデータの描写
    #-----------------------------------
    """
    # 先頭の 0~9 のラベルの画像データを plot
    # plt.subplots(...) から,
    # Figure クラスのオブジェクト、Axis クラスのオブジェクト作成
    figure, axis = plt.subplots( 
                       nrows = 2, ncols = 5,
                       sharex = True, sharey = True     # x,y 軸をシャアする
                   )
    # 2 × 5 配列を１次元に変換
    axis = axis.flatten()
    # 数字の 0~9 の plot 用の for ループ
    for i in range(10):
        image = X_train[y_train == i][0]    #
        image = image.reshape(28,28)        # １次元配列を shape = [28 ,28] に reshape
        axis[i].imshow(
            image,
            cmap = "Greys",
            interpolation = "nearest"   # 補間方法
        )
    axis[0].set_xticks( [] )
    axis[0].set_yticks( [] )
    plt.tight_layout()
    plt.show()
    """

    #======================================================================
    # DataLoader への変換
    #======================================================================
    #---------------------------------------------------------------
    # numpy データを Tensor に変換
    # Pytorchでは入力データがfloat型、target（ラベル）はlong型という決まり
    # Tensor型 : 
    #---------------------------------------------------------------
    X_train = torch.Tensor( X_train )           # float 型の FloatTensor
    X_test = torch.Tensor( X_test )
    y_train = torch.LongTensor( y_train )       # 整数型の Tensor
    y_test = torch.LongTensor( y_test )         #

    print( "X_train <Tensor> :", X_train )
    print( "X_test <Tensor> :", X_test )
    print( "y_train <Tensor> :", y_train )
    print( "y_test <Tensor> :", y_test )

    #---------------------------------------------------------------
    # 微分可能にしたければ、更に Tensor を Variable に変換
    # Variable.data : Tensor
    # Variable.grad : 勾配情報
    #---------------------------------------------------------------
    pass    # 今回は変換しない

    #---------------------------------------------------------------
    # data と label をセットにした Dataset の作成
    #---------------------------------------------------------------
    ds_train = TensorDataset( X_train, y_train )
    ds_test = TensorDataset( X_test, y_test )

    print( "ds_train :", ds_train )
    print( "ds_test :", ds_test )

    #---------------------------------------------------------------
    # Dataset を DataLoader に変換
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

    # dloader_train.datset
    # dloader_train.sampler = <RandomSampler, len() = 60000>
    # dloader_train[0] = [<Tensor, len() = 32>, <Tensor, len() = 32>]
    # dloader_train[1874] = [<Tensor, len() = 32>, <Tensor, len() = 32>]
    print( "dloader_train :", dloader_train )
    print( "dloader_test :", dloader_test )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    #---------------------------------------------------------------
    # nn.Sequential で Keras 風に 簡単にレイヤーを追加
    # この例では、多層パーセプトロン( 784→100→100→10)
    #---------------------------------------------------------------
    model = nn.Sequential()
    model.add_module( name = "fc1", module = nn.Linear( in_features = 28*28*1, out_features = 100 ) )
    model.add_module( "relu1", nn.ReLU() )
    model.add_module( "fc2", nn.Linear( 100, 100 ) )
    model.add_module( "relu2", nn.ReLU() )
    model.add_module( "fc3", nn.Linear( 100, NUM_CLASSES ) )

    print( "model :\n", model )

    #---------------------------------------------------------------
    # 損失関数を設定
    #---------------------------------------------------------------
    loss_fn = nn.CrossEntropyLoss()

    #---------------------------------------------------------------
    # optimizer を設定
    #---------------------------------------------------------------
    optimizer = torch.optim.Adam(
        params = model.parameters(),    # 勾配の計算対象となる model のパラメーター
        lr = LEARNING_RATE
    )

    #======================================================================
    # モデルの学習フェイズ
    #======================================================================
    # model を学習モードに切り替える（PyTorch特有の処理）
    model.train()

    # for ループでエポック数分トレーニング
    for epoch in range( NUM_EPOCHES ):

        # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
        for (inputs,targets) in dloader_train:
            # 勾配を 0 に初期化（この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
            optimizer.zero_grad()

            # 学習用データをモデルに流し込む
            # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
            # outputs : Tensor型
            outputs = model( inputs )
            #print( "outputs :", outputs )   # shape = [32,10]

            # 出力と教師データを損失関数に設定し、誤差 loss を計算
            # この設定は、損失関数を __call__ をオーバライト
            # lossはPytorchのVariableとして帰ってくるので、これをloss.data[0]で数値として見る必要があり
            loss = loss_fn( outputs, targets )

            # 誤差逆伝搬
            loss.backward()

            # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
            optimizer.step()

            #
            print( "epoch %d / loss = %0.3f" % ( epoch + 1, loss ) )

    #======================================================================
    # モデルの推論フェーズ
    #======================================================================
    n_correct = 0

    # model を推論モードに切り替える（PyTorch特有の処理）
    model.eval()

    # torch.no_grad()
    # 微分を行わない処理の範囲を with 構文で囲む
    # pytorchではtrain時，forward計算時に勾配計算用のパラメータを保存しておくことでbackward計算の高速化を行っており、
    # これは，model.eval()で行っていてもパラメータが保存されるために、torch.no_grad() でパラメータの保存を止める必要がある。
    with torch.no_grad():
        for (inputs,targets) in dloader_test:
            # テストデータをモデルに流し込む
            outputs = model( inputs )

            # 確率値が最大のラベル 0~9 を予想ラベルとする。
            # dim = 1 ⇒ 列方向で最大値をとる
            # Returns : (Tensor, LongTensor)
            _, predicts = torch.max( outputs.data, dim = 1 )
            #print( "predicts :", predicts )

            #--------------------
            # 正解数のカウント
            #--------------------
            # Tensorのサイズを変えたい場合にはviewメソッドを使います（numpyでのreshapeに対応）。
            #print( "targets.data.view_as() :", targets.data.view_as( predicts ) )
            
            # ? ミニバッチ内で一致したラベルをカウント
            n_correct += predicts.eq( targets.data.view_as( predicts ) ).sum()

    # 正解率
    n_tests = len( dloader_test.dataset )
    print( "Accuracy [test] : {}/{} {:.0f}%\n".format( n_correct, n_tests, 100 * n_correct / n_tests ) )

    print("Finish main()")
    return

    
if __name__ == '__main__':
     main()