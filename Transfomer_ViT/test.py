import os
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
import cv2

# sklearn
from sklearn.model_selection import train_test_split

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

# 自作モジュール
from data.dataset import TempleteDataset, TempleteDataLoader
from models.networks import TempleteNetworks
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="dataset/templete_dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--n_samplings', type=int, default=100000, help="サンプリング最大数")
    parser.add_argument('--batch_size_test', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=128, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=128, help="入力画像の幅（pixel単位）")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
    parser.add_argument('--detect_nan', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    # 出力フォルダの作成
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name, "output" ) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name, "output"))

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

    # seed 値の固定
    if( args.use_cuda_deterministic ):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # NAN 値の検出
    if( args.detect_nan ):
        torch.autograd.set_detect_anomaly(True)

    # tensorboard 出力
    board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    ds_test = TempleteDataset( args, args.dataset_dir, datamode = "test", image_height = args.image_height, image_width = args.image_width, data_augument = False, debug = args.debug )
    dloader_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size_test, shuffle = False, num_workers = args.n_workers, pin_memory = True )

    #================================
    # モデルの構造を定義する。
    #================================
    model_G = TempleteNetworks().to(device)
    if( args.debug ):
        print( "model_G\n", model_G )

    # モデルを読み込む
    if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
        load_checkpoint(model_G, device, args.load_checkpoints_path )
        
    #================================
    # モデルの推論
    #================================    
    print("Starting Testing Loop...")
    n_print = 1
    model_G.eval()
    for step, inputs in enumerate( tqdm( dloader_test, desc = "Samplings" ) ):
        if inputs["image"].shape[0] != args.batch_size_test:
            break

        # ミニバッチデータを GPU へ転送
        image_name = inputs["image_name"]
        image = inputs["image"].to(device)
        if( args.debug and n_print > 0):
            print( "image.shape : ", image.shape )

        #----------------------------------------------------
        # 生成器の推論処理
        #----------------------------------------------------
        with torch.no_grad():
            output = model_G( image )
            if( args.debug and n_print > 0 ):
                print( "output.shape : ", output.shape )

        #====================================================
        # 推論結果の保存
        #====================================================
        save_image_w_norm( output, os.path.join( args.results_dir, args.exper_name, "output", image_name[0] ) )

        # tensorboard
        visuals = [
            [ image, output ],
        ]
        board_add_images(board_test, 'test', visuals, step+1)

        n_print -= 1
        if( step >= args.n_samplings ):
            break
