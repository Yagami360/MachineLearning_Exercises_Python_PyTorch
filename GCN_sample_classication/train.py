import os
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
import cv2

from sklearn.model_selection import train_test_split

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

# 自作モジュール
from dataset import load_dataset
from networks import GraphConvolutionNetworks
from utils import save_checkpoint, load_checkpoint
from utils import calc_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="gcn_for_classication", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/cora_dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=200, help="エポック数")    
    #parser.add_argument('--batch_size', type=int, default=32, help="バッチサイズ")
    #parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    #parser.add_argument('--batch_size_test', type=int, default=1, help="バッチサイズ")
    parser.add_argument("--n_classes", type=int, default=7, help="クラス数")
    parser.add_argument('--lr', type=float, default=0.01, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")

    parser.add_argument("--n_diaplay_step", type=int, default=10,)
    parser.add_argument('--n_display_valid_step', type=int, default=10, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=200,)

    parser.add_argument("--val_rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
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
    if not( os.path.exists(args.save_checkpoints_dir) ):
        os.mkdir(args.save_checkpoints_dir)
    if not( os.path.exists(os.path.join(args.save_checkpoints_dir, args.exper_name)) ):
        os.mkdir( os.path.join(args.save_checkpoints_dir, args.exper_name) )

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

    # tensorboard 出力
    board_train = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name) )
    board_valid = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_valid") )
    #board_test = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_test") )

    #================================
    # データセットの読み込み
    #================================    
    features, labels, adj_matrix = load_dataset( dataset_dir = args.dataset_dir, device = device )
    if( args.debug ):
        print( "features.shape : ", features.shape )
        print( "labels.shape : ", labels.shape )
        print( "adj_matrix.shape : ", adj_matrix.shape )

    # データセットの分類
    train_valid_idx = range(0, 2000)
    test_idx = range(2001, 2707)
    train_idx, valid_idx = train_test_split(train_valid_idx, test_size=args.val_rate, random_state=args.seed, stratify=labels[train_valid_idx].cpu().numpy())

    #================================
    # モデルの構造を定義する。
    #================================
    model = GraphConvolutionNetworks( n_inputs = features.shape[1], n_outputs = args.n_classes ).to(device)
    if( args.debug ):
        print( "model\n", model )

    # モデルを読み込む
    if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
        load_checkpoint(model, device, args.load_checkpoints_path )
        
    #================================
    # optimizer の設定
    #================================
    optimizer = optim.Adam( params = model.parameters(), lr = args.lr, betas = (args.beta1,args.beta2) )

    #================================
    # loss 関数の設定
    #================================
    loss_fn = nn.NLLLoss()  # The negative log likelihood loss

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    n_print = 1
    step = 0
    for epoch in tqdm( range(args.n_epoches), desc = "epoches" ):
        model.train()

        # forword 処理 / output : 分類結果（各ベクトル値の値が分類の確率値）softmax 出力
        output = model( features, adj_matrix )
        if( args.debug and n_print > 0 ):
            print( "output.shape : ", output.shape )

        # 損失関数を計算する
        loss = loss_fn( output[train_idx], labels[train_idx] )

        # ネットワークの更新処理
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #====================================================
        # 学習過程の表示
        #====================================================
        if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
            # 正解率の計算
            accuracy = calc_accuracy( output[train_idx], labels[train_idx] )
            print( "[train] step={}, loss={:.5f}, accuracy={:.5f}".format(step, loss.item(), accuracy.item()) )

            # tensorboard 出力
            board_train.add_scalar('G/loss', loss.item(), step)
            board_train.add_scalar('G/accuracy', accuracy.item(), step)

        if( step == 0 or ( step % args.n_display_valid_step == 0 ) ):
            loss = loss_fn( output[valid_idx], labels[valid_idx] )

            # 正解率の計算
            accuracy = calc_accuracy( output[valid_idx], labels[valid_idx] )
            print( "[valid] step={}, loss={:.5f}, accuracy={:.5f}".format(step, loss.item(), accuracy.item()) )

            # tensorboard 出力
            board_valid.add_scalar('G/loss', loss.item(), step)
            board_valid.add_scalar('G/accuracy', accuracy.item(), step)

        step += 1
        n_print -= 1

    print("Finished Training Loop.")
    save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth') )