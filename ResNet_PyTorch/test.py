# -*- coding:utf-8 -*-
import argparse
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
from PIL import Image

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn

import torchvision      # 画像処理関連
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 自作クラス
from networks import ResNet18
from utils import save_checkpoint, load_checkpoint
from utils import board_add_image, board_add_images

if __name__ == '__main__':
    """
    ResNet-18 によるクラス分類の推論処理
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="ResNet18_test", help="実験名")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    #parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU') 
    parser.add_argument('--dataset', choices=['mnist', 'cifar-10'], default="mnist", help="データセットの種類")
    parser.add_argument('--dataset_dir', type=str, default="dataset", help="データセットのディレクトリ")
    parser.add_argument('--results_dir', type=str, default="results", help="生成画像の出力ディレクトリ")
    parser.add_argument('--load_checkpoints_dir', type=str, default="", help="モデルの読み込みディレクトリ")
    parser.add_argument('--n_samplings', type=int, default=100, help="サンプリング数")
    parser.add_argument('--batch_size', type=int, default=64, help="バッチサイズ")
    parser.add_argument('--image_size', type=int, default=224, help="入力画像のサイズ（pixel単位）")
    parser.add_argument('--n_fmaps', type=int, default=64, help="１層目の特徴マップの枚数")
    parser.add_argument("--n_classes", type = int, default = 10 )
    parser.add_argument("--seed", type=int, default=8, help="乱数シード値")
    parser.add_argument('--debug', action='store_true', help="デバッグモード有効化")
    args = parser.parse_args()

    # 実行条件の出力
    print( "----------------------------------------------" )
    print( "実行条件" )
    print( "----------------------------------------------" )
    print( "開始時間：", datetime.now() )
    print( "PyTorch version :", torch.__version__ )
    for key, value in vars(args).items():
        print('%s: %s' % (str(key), str(value)))

    # 実行 Device の設定
    if( args.device == "gpu" ):
        use_cuda = torch.cuda.is_available()
        if( use_cuda == True ):
            device = torch.device( "cuda" )
            #torch.cuda.set_device(args.gpu_ids[0])
            print( "実行デバイス :", device)
            print( "GPU名 :", torch.cuda.get_device_name(device))
            print("torch.cuda.current_device() =", torch.cuda.current_device())
        else:
            print( "can't using gpu." )
            device = torch.device( "cpu" )
            print( "実行デバイス :", device)
    else:
        device = torch.device( "cpu" )
        print( "実行デバイス :", device)

    print('-------------- End ----------------------------')

    # 各種出力ディレクトリ
    if not( os.path.exists(args.results_dir) ):
        os.mkdir(args.results_dir)
    if not( os.path.exists(os.path.join(args.results_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.results_dir, args.exper_name) )

    # seed 値の固定
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    #======================================================================
    # データセットを読み込み or 生成
    # データの前処理
    #======================================================================
    if( args.dataset == "mnist" ):
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=Image.LANCZOS ),
                transforms.ToTensor(),   # Tensor に変換]
                transforms.Normalize((0.5,), (0.5,)),   # 1 channel 分
            ]
        )

        ds_test = torchvision.datasets.MNIST(
            root = args.dataset_dir,
            train = False,
            transform = transform,
            target_transform = None,
            download = True
        )
    elif( args.dataset == "cifar-10" ):
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=Image.LANCZOS ),
                transforms.ToTensor(),   # Tensor に変換
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        ds_test = torchvision.datasets.CIFAR10(
            root = args.dataset_dir,
            train = False,
            transform = transform,
            target_transform = None,
            download = True
        )
    else:
        raise NotImplementedError('dataset %s not implemented' % args.dataset)

    # TensorDataset → DataLoader への変換        
    dloader_test = DataLoader(
        dataset = ds_test,
        batch_size = args.batch_size,
        shuffle = False
    )

    if( args.debug ):
        print( "ds_test :\n", ds_test )

    #======================================================================
    # モデルの構造を定義する。
    #======================================================================
    if( args.dataset == "mnist" ):
        model = ResNet18(
                n_in_channels = 1,
                n_fmaps = args.n_fmaps,
                n_classes = args.n_classes
        ).to(device)
    else:
        model = ResNet18(
                n_in_channels = 3,
                n_fmaps = args.n_fmaps,
                n_classes = args.n_classes
        ).to(device)

    if( args.debug ):
        print( "model :\n", model )

    # モデルを読み込む
    if not args.load_checkpoints_dir == '' and os.path.exists(args.load_checkpoints_dir):
        init_step = load_checkpoint(model, device, os.path.join(args.load_checkpoints_dir, "model_final.pth") )

    #======================================================================
    # モデルの推論処理
    #======================================================================
    print("Starting Test Loop...")
    n_tests_total = 0
    n_correct_total = 0
    n_print = 1
    model.eval()
    # DataLoader から 1minibatch 分取り出し、ミニバッチ処理
    for step, (images,targets) in enumerate( tqdm( dloader_test, desc = "Samplings" ) ):
        # ミニバッチデータを GPU へ転送
        images = images.to( device )
        targets = targets.to( device )

        #====================================================
        # モデル の 推論処理
        #====================================================
        with torch.no_grad():
            output = model( images )
            if( args.debug and n_print > 0 ):
                print( "output.shape :", output.shape )
                
        #----------------------------------------------------
        # 正解率を計算する。（バッチデータ）
        #----------------------------------------------------
        # 確率値が最大のラベル 0~9 を予想ラベルとする。
        # dim = 1 ⇒ 列方向で最大値をとる
        # Returns : (Tensor, LongTensor)
        _, predicts = torch.max( output.data, dim = 1 )
        if( args.debug and n_print > 0 ):
            print( "predicts.shape :", predicts.shape )

        # 正解数のカウント
        n_tests = targets.size(0)
        n_tests_total += n_tests

        # ミニバッチ内で一致したラベルをカウント
        n_correct = ( predicts == targets ).sum().item()
        n_correct_total += n_correct

        accuracy = n_correct / n_tests
        print( "step={}, accuracy={:.5f}".format(step, accuracy) )

        n_print -= 1
        if( step >= args.n_samplings ):
            break

    accuracy_total = n_correct_total / n_tests_total
    print( "accuracy_total={:.5f}".format(accuracy_total) )

    print("Finished Test Loop.")