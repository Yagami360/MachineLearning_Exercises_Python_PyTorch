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
from tensorboardX import SummaryWriter

# 自作モジュール
from dataset import CIHPDataset, CIHPDataLoader
from models.graphonomy import GraphonomyIntraGraphReasoning
from models.graph_params import get_graph_adj_matrix
from models.losses import ParsingCrossEntropyLoss, CrossEntropy2DLoss
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm

if __name__ == '__main__':
    """
    Graphonomy の１つのデータセットに対する Intra-Graph Reasoning での学習処理
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="graphonomy_intra_graph_reasoning", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/CIHP_4w")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=100, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=2, help="バッチサイズ")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--batch_size_test', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=512, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=512, help="入力画像の幅（pixel単位）")
    parser.add_argument("--n_classes", type=int, default=20, help="グラフ構造のクラス数")
    parser.add_argument("--n_node_features", type=int, default=128, help="グラフの各頂点の特徴次元")
    parser.add_argument('--lr', type=float, default=0.007, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=10,)
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument('--flip', action='store_true')

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

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    ds_train = CIHPDataset( args, args.dataset_dir, datamode = "train", flip = args.flip, data_augument = args.data_augument, debug = args.debug )

    # 学習用データセットとテスト用データセットの設定
    index = np.arange(len(ds_train))
    train_index, valid_index = train_test_split( index, test_size=args.val_rate, random_state=args.seed )
    if( args.debug ):
        print( "train_index.shape : ", train_index.shape )
        print( "valid_index.shape : ", valid_index.shape )
        print( "train_index[0:10] : ", train_index[0:10] )
        print( "valid_index[0:10] : ", valid_index[0:10] )

    dloader_train = torch.utils.data.DataLoader(Subset(ds_train, train_index), batch_size=args.batch_size, shuffle=True, num_workers = args.n_workers, pin_memory = True )
    dloader_valid = torch.utils.data.DataLoader(Subset(ds_train, valid_index), batch_size=args.batch_size_valid, shuffle=False, num_workers = args.n_workers, pin_memory = True )

    #================================
    # モデルの構造を定義する。
    #================================
    model = GraphonomyIntraGraphReasoning( n_in_channels = 3, n_classes = args.n_classes, n_node_features = args.n_node_features ).to(device)
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
    loss_fn = ParsingCrossEntropyLoss()
    #loss_fn = CrossEntropy2DLoss(device)

    #================================
    # 定義済みグラフ構造の取得
    #================================
    adj_matrix_cihp_to_cihp, adj_matrix_pascal_to_pascal, adj_matrix_cihp_to_pascal = get_graph_adj_matrix()
    adj_matrix_cihp_to_cihp, adj_matrix_pascal_to_pascal, adj_matrix_cihp_to_pascal = adj_matrix_cihp_to_cihp.to(device), adj_matrix_pascal_to_pascal.to(device), adj_matrix_cihp_to_pascal.to(device)

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    n_print = 1
    step = 0
    for epoch in tqdm( range(args.n_epoches), desc = "epoches" ):
        for iter, inputs in enumerate( tqdm( dloader_train, desc = "iters" ) ):
            model.train()

            # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
            if inputs["image"].shape[0] != args.batch_size:
                break

            # ミニバッチデータを GPU へ転送
            image = inputs["image"].to(device)
            target = inputs["target"].to(device)
            if( args.debug and n_print > 0):
                print( "image.shape : ", image.shape )
                print( "target.shape : ", target.shape )
                print( "adj_matrix_cihp_to_cihp.shape : ", adj_matrix_cihp_to_cihp.shape )

            # forword 処理
            output, embedded, graph, reproj_feature = model( image, adj_matrix_cihp_to_cihp )
            if( args.debug and n_print > 0 ):
                print( "output.shape : ", output.shape )

            # 損失関数を計算する
            loss = loss_fn( output, target )
            #loss = torch.zeros(1, requires_grad=True).float().to(device)

            # ネットワークの更新処理
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                # loss
                board_train.add_scalar('G/loss', loss.item(), step)
                print( "step={}, loss={:.5f}".format(step, loss.item()) )

                # visual images
                visuals = [
                    [ image, target ],
                    [ output[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes//4) ],
                    [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//4 + 1, args.n_classes//2) ],
                    [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + 1, args.n_classes//2 + args.n_classes//4) ],
                    [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + args.n_classes//4 + 1, args.n_classes) ],
                ]
                board_add_images(board_train, 'train', visuals, step+1)

                # visual deeplab v3+ output
                visuals = [
                    [ embedded[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes//4) ],
                    [ embedded[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//4 + 1, args.n_classes//2) ],
                    [ embedded[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + 1, args.n_classes//2 + args.n_classes//4) ],
                    [ embedded[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + args.n_classes//4 + 1, args.n_classes) ],
                ]
                board_add_images(board_train, 'train_deeplab_embedded', visuals, step+1)

                # visual graph output
                visuals = [
                    [ graph.transpose(1,0) ],
                ]
                board_add_images(board_train, 'train_graph', visuals, step+1)

                # visual feature output
                visuals = [
                    [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes//4) ],
                    [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//4 + 1, args.n_classes//2) ],
                    [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + 1, args.n_classes//2 + args.n_classes//4) ],
                    [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + args.n_classes//4 + 1, args.n_classes) ],
                ]
                board_add_images(board_train, 'train_re-proj_feature', visuals, step+1)

            if( step == 0 or ( step % args.n_display_valid_step == 0 ) ):
                loss_total = 0
                n_valid_loop = 0
                for iter, inputs in enumerate( tqdm(dloader_valid, desc = "eval iters") ):
                    model.eval()            

                    # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                    if inputs["image"].shape[0] != args.batch_size_valid:
                        break

                    # ミニバッチデータを GPU へ転送
                    image = inputs["image"].to(device)
                    target = inputs["target"].to(device)

                    # 推論処理
                    with torch.no_grad():
                        output, embedded, graph, reproj_feature = model( image, adj_matrix_cihp_to_cihp )

                    # 損失関数を計算する
                    loss = loss_fn( output, target )
                    loss_total += loss

                    # 生成画像表示
                    if( iter <= args.n_display_valid ):
                        # visual images
                        visuals = [
                            [ image, target ],
                            [ output[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes//4) ],
                            [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//4 + 1, args.n_classes//2) ],
                            [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + 1, args.n_classes//2 + args.n_classes//4) ],
                            [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + args.n_classes//4 + 1, args.n_classes) ],
                        ]
                        board_add_images(board_valid, 'valid/{}'.format(iter), visuals, step+1)

                        # visual deeplab v3+ output
                        visuals = [
                            [ embedded[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes//4) ],
                            [ embedded[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//4 + 1, args.n_classes//2) ],
                            [ embedded[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + 1, args.n_classes//2 + args.n_classes//4) ],
                            [ embedded[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + args.n_classes//4 + 1, args.n_classes) ],
                        ]
                        board_add_images(board_train, 'valid_deeplab_embedded/{}'.format(iter), visuals, step+1)

                        # visual graph output
                        visuals = [
                            [ graph.transpose(1,0) ],
                        ]
                        board_add_images(board_train, 'valid_graph/{}'.format(iter), visuals, step+1)

                        # visual feature output
                        visuals = [
                            [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes//4) ],
                            [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//4 + 1, args.n_classes//2) ],
                            [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + 1, args.n_classes//2 + args.n_classes//4) ],
                            [ reproj_feature[:,i,:,:].unsqueeze(1) for i in range(args.n_classes//2 + args.n_classes//4 + 1, args.n_classes) ],
                        ]
                        board_add_images(board_train, 'valid_re-proj_feature/{}'.format(iter), visuals, step+1)

                    n_valid_loop += 1

                # loss 値表示
                board_valid.add_scalar('G/loss', loss_total.item()/n_valid_loop, step)

            #====================================================
            # モデルの保存
            #====================================================
            if( epoch % args.n_save_epoches == 0 ):
                save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_ep%03d.pth' % (epoch)) )
                save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth') )
                print( "saved checkpoints" )
                
            step += 1
            n_print -= 1

    print("Finished Training Loop.")
    save_checkpoint( model, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth') )
