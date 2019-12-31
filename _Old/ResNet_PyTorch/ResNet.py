# -*- coding:utf-8 -*-
import os
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import tensorboardX as tbx


class BasicBlock( nn.Module ):
    """
    """
    def __init__( 
        self,
        device,
        n_in_channels = 3,
        n_out_channels = 3,
        stride = 1,
    ):
        """
        [Args]
            n_in_channels : <int> 入力画像のチャンネル数
            n_out_channels : <int> 出力画像のチャンネル数
            stride : <int>
        """
        super( BasicBlock, self ).__init__()
        self._device = device

        self._layer1 = nn.Sequential(
            nn.Conv2d( n_in_channels, n_out_channels, kernel_size=3, stride=stride, padding=1 ),
            nn.BatchNorm2d( n_out_channels ),
            nn.LeakyReLU( 0.2, inplace=True ),
        ).to( self._device )

        self._layer2 = nn.Sequential(
            nn.Conv2d( n_out_channels, n_out_channels, kernel_size=3, stride=1, padding=1 ),
            nn.BatchNorm2d( n_out_channels ),
        ).to( self._device )

        # shortcut connection は、恒等写像
        self._shortcut_connections = nn.Sequential()

        # 入出力次元が異なる場合は、ゼロパディングで、次元の不一致箇所を０で埋める。
        if( n_in_channels != n_out_channels ):
            self._shortcut_connections = nn.Sequential(
                nn.Conv2d( n_in_channels, n_out_channels, kernel_size=1, stride=stride, padding=0,bias=False),
                nn.BatchNorm2d( n_out_channels )
            ).to( self._device )

        return

    def forward( self, x ):
        out = self._layer1(x)
        out = self._layer2(out)

        # shortcut connection からの経路を加算
        out += self._shortcut_connections(x)
        return out


class ResNet18( nn.Module ):
    def __init__( 
        self,
        device,
        n_in_channels = 3,
        n_classes = 10
    ):
        super( ResNet18, self ).__init__()
        
        self._device = device
        self._layer0 = nn.Sequential(
            nn.Conv2d( n_in_channels, 64, kernel_size=7, stride=2, padding=3 ),
            nn.BatchNorm2d( 64 ),
            nn.LeakyReLU( 0.2, inplace=True ),
            nn.MaxPool2d( kernel_size=3, stride=2, padding=1 )
        ).to( self._device )
        
        self._layer1 = nn.Sequential(
                BasicBlock(
                    device = self._device,
                    n_in_channels = 64, n_out_channels = 64, stride = 1
                ),
                BasicBlock(
                    device = self._device,
                    n_in_channels = 64, n_out_channels = 64, stride = 1
                ),          
        )

        self._layer2 = nn.Sequential(
                BasicBlock(
                    device = self._device,
                    n_in_channels = 64, n_out_channels = 128, stride = 2
                ),
                BasicBlock(
                    device = self._device,
                    n_in_channels = 128, n_out_channels = 128, stride = 1
                ),          
        )

        self._layer3 = nn.Sequential(
                BasicBlock(
                    device = self._device,
                    n_in_channels = 128, n_out_channels = 256, stride = 2
                ),
                BasicBlock(
                    device = self._device,
                    n_in_channels = 256, n_out_channels = 256, stride = 1
                ),          
        )

        self._layer4 = nn.Sequential(
                BasicBlock(
                    device = self._device,
                    n_in_channels = 256, n_out_channels = 512, stride = 2
                ),
                BasicBlock(
                    device = self._device,
                    n_in_channels = 512, n_out_channels = 512, stride = 1
                ),          
        )

        self._avgpool = nn.AvgPool2d( 7, stride=1 ).to( self._device )
        self._fc_layer = nn.Linear( 512, n_classes ).to( self._device )
        return

    def forward( self, x ):
        out = self._layer0(x)
        out = self._layer1(out)
        out = self._layer2(out)
        out = self._layer3(out)
        out = self._layer4(out)
        #out = torch.squeeze(out)
        out = self._avgpool(out)
        out = out.view( out.size(0), -1 )
        out = self._fc_layer(out)
        return out


class ResNetClassifier( object ):
    """
    [public]
    [protected] 変数名の前にアンダースコア _ を付ける
        _device : <toech.cuda.device> 実行デバイス
        _n_classes : 
        _n_epoches : <int> エポック数（学習回数）
        _batch_size : <int> ミニバッチ学習時のバッチサイズ
        _learnig_rate : <float> 最適化アルゴリズムの学習率

        _model : <nn.Module> ResNet のネットワークモデル
        _loss_fn : 損失関数
        _optimizer : <torch.optim.Optimizer> モデルの最適化アルゴリズム
        _loss_historys : <list> 損失関数値の履歴（イテレーション毎）

    [private] 変数名の前にダブルアンダースコア __ を付ける（Pythonルール）
    """
    def __init__(
        self,
        device,
        n_classes = 10,
        n_epoches = 10,
        batch_size = 64,
        learing_rate = 0.0001,
    ):
        self._device = device
        self._n_classes = n_classes
        self._n_epoches = n_epoches
        self._batch_size = batch_size
        self._learning_rate = learing_rate

        self._loss_fn = None
        self._optimizer = None
        self._loss_historys = []

        self.model()
        self.loss()
        self.optimizer()
        return

    def print( self, str = "" ):
        print( "----------------------------------" )
        print( "ResNetClassifier" )
        print( self )
        print( str )
        print( "_device :", self._device )
        print( "_n_classes :", self._n_classes )        
        print( "_n_epoches :", self._n_epoches )
        print( "_batch_size :", self._batch_size )
        print( "_learning_rate :", self._learning_rate )
        print( "_model :", self._model )
        print( "_loss_fn :", self._loss_fn )
        print( "_optimizer :", self._optimizer )
        print( "----------------------------------" )
        return

    @property
    def loss_history( self ):
        return self._loss_historys

    def model( self ):
        """
        モデルの定義を行う。
        [Args]
        [Returns]
        """
        self._model = ResNet18(
            device = self._device,
            n_in_channels = 3,
            n_classes = self._n_classes
        )
        return

    def loss( self ):
        """
        損失関数の設定を行う。
        [Args]
        [Returns]
        """
        self._loss_fn = nn.CrossEntropyLoss()
        return

    def optimizer( self ):
        """
        モデルの最適化アルゴリズムの設定を行う。
        [Args]
        [Returns]
        """
        self._optimizer = optim.Adam(
            params = self._model.parameters(),
            lr = self._learning_rate,
            betas = (0.5,0.999)
        )
        return

    def fit( self, dloader, n_sava_step = 5, result_path = "./result" ):
        """
        指定されたトレーニングデータで、モデルの fitting 処理を行う。
        [Args]
            dloader : <DataLoader> 学習用データセットの DataLoader
            n_sava_step : <int> 学習途中での生成画像の保存間隔（イテレーション単位）
            result_path : <str> 学習途中＆結果を保存するディレクトリ
        [Returns]
        """
        if( os.path.exists( result_path ) == False ):
            os.mkdir( result_path )

        # TensorBoard の Writter
        writer = tbx.SummaryWriter( result_path )

        #-------------------------------------
        # モデルを学習モードに切り替える。
        #-------------------------------------
        self._model.train()

        #-------------------------------------
        # 学習処理ループ
        #-------------------------------------
        iterations = 0      # 学習処理のイテレーション回数

        print("Starting Training Loop...")
        # エポック数分トレーニング
        for epoch in tqdm( range(self._n_epoches), desc = "Epoches" ):
            # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
            for (images,targets) in tqdm( dloader, desc = "minibatch process in DataLoader" ):
                #print( "images.size() : ", images.size() )
                #print( "targets.size() : ", targets.size() )

                # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する
                # （後の計算で、shape の不一致をおこすため）
                if images.size()[0] != self._batch_size:
                    break

                iterations += 1

                # ミニバッチデータを GPU へ転送
                images = images.to( self._device )
                targets = targets.to( self._device )

                #----------------------------------------------------
                # 勾配を 0 に初期化
                # （この初期化処理が必要なのは、勾配がイテレーション毎に加算される仕様のため）
                #----------------------------------------------------
                self._optimizer.zero_grad()

                #----------------------------------------------------
                # 学習用データをモデルに流し込む
                # model(引数) で呼び出せるのは、__call__ をオーバライトしているため
                #----------------------------------------------------
                output = self._model( images )

                #----------------------------------------------------
                # 損失関数を計算する
                #----------------------------------------------------
                loss = self._loss_fn( output, targets )
                self._loss_historys.append( loss.item() )

                #----------------------------------------------------
                # 誤差逆伝搬
                #----------------------------------------------------
                loss.backward()

                #----------------------------------------------------
                # backward() で計算した勾配を元に、設定した optimizer に従って、重みを更新
                #----------------------------------------------------
                self._optimizer.step()

                # 学習経過の表示処理
                writer.add_scalar( "data/loss", loss.item(), iterations )
                print( "\nepoch = %d / iterations = %d / loss = %f" % ( epoch, iterations, loss ) )

            # エポック度の処理
            self.save_model()
                
        self.save_model()
        writer.export_scalars_to_json( os.path.join( result_path, "tensorboard.json" ) )
        writer.close()
        print("Finished Training Loop.")
        return

    def predict( self, dloader ):
        """
        fitting 処理したモデルで推定を行い、予想クラスラベル値を返す。

        [Args]
            dloader : <DataLoader> テスト用データセットの DataLoader
        [Returns]
            predicts : <Tensor> 予想クラスラベル
        """
        # model を推論モードに切り替える（PyTorch特有の処理）
        self._model.eval()

        # torch.no_grad()
        # 微分を行わない処理の範囲を with 構文で囲む
        # pytorchではtrain時，forward計算時に勾配計算用のパラメータを保存しておくことでbackward計算の高速化を行っており、
        # これは，model.eval()で行っていてもパラメータが保存されるために、torch.no_grad() でパラメータの保存を止める必要がある。
        with torch.no_grad():
            for (inputs,targets) in dloader:
                # ミニバッチデータを GPU へ転送
                inputs = inputs.to( self._device )
                targets = targets.to( self._device )

                # テストデータをモデルに流し込む
                outputs = self._model( inputs )

                # 確率値が最大のラベル 0~9 を予想ラベルとする。
                # dim = 1 ⇒ 列方向で最大値をとる
                # Returns : (Tensor, LongTensor)
                _, predicts = torch.max( outputs.data, dim = 1 )
                #print( "predicts :", predicts )

        return predicts


    def accuracy( self, dloader ):
        """
        指定したデータでの正解率 [accuracy] を計算する。

        [Args]
            dloader : <DataLoader> テスト用データセットの DataLoader
        [Returns]
            accuracy : <float> 正解率

        """
        n_correct = 0
        n_tests = 0

        # torch.no_grad()
        # 微分を行わない処理の範囲を with 構文で囲む
        # pytorchではtrain時，forward計算時に勾配計算用のパラメータを保存しておくことでbackward計算の高速化を行っており、
        # これは，model.eval()で行っていてもパラメータが保存されるために、torch.no_grad() でパラメータの保存を止める必要がある。
        with torch.no_grad():
            for (inputs,targets) in dloader:
                # ミニバッチデータを GPU へ転送
                inputs = inputs.to( self._device )
                targets = targets.to( self._device )

                # テストデータをモデルに流し込む
                outputs = self._model( inputs )

                # 確率値が最大のラベル 0~9 を予想ラベルとする。
                # dim = 1 ⇒ 列方向で最大値をとる
                # Returns : (Tensor, LongTensor)
                _, predicts = torch.max( outputs.data, dim = 1 )
                #print( "predicts :", predicts )

                #--------------------
                # 正解数のカウント
                #--------------------
                n_tests += targets.size(0)

                # ミニバッチ内で一致したラベルをカウント
                n_correct += ( predicts == targets ).sum().item()

        # 正解率の計算
        accuracy = n_correct / n_tests
        print( "Accuracy [test] : {}/{} {:.5f}\n".format( n_correct, n_tests, accuracy ) )

        return accuracy


    def save_model( self, save_dir = "./checkpoint" ):
        if( os.path.exists( save_dir ) == False ):
            os.mkdir( save_dir )

        torch.save( 
            self._model.state_dict(), 
            os.path.join( save_dir, "model.pth" )
        )
        return

    def load_model( self, load_dir = "./checkpoint" ):
        self._model.load_state_dict( 
            torch.load( os.path.join( load_dir, "model.pth" ) )
        )
        return