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
from models.discriminators import PatchGANDiscriminator
from models.losses import ParsingCrossEntropyLoss, CrossEntropy2DLoss, VGGLoss, LSGANLoss
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm
from utils.decode_labels import decode_labels_tsr

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
    parser.add_argument("--n_output_channels", type=int, default=20, help="出力データのチャンネル次元（通常クラス数）")
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

    parser.add_argument('--lambda_l1', type=float, default=5.0, help="L1損失関数の係数値")
    parser.add_argument('--lambda_vgg', type=float, default=5.0, help="VGG perceptual loss_G の係数値")
    parser.add_argument('--lambda_entropy', type=float, default=1.0, help="クロスエントロピー損失関数の係数値")
    parser.add_argument('--lambda_adv', type=float, default=1.0, help="Adv loss_G の係数値")

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

    # NAN 値の検出
    if( args.detect_nan ):
        torch.autograd.set_detect_anomaly(True)

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
    model_G = GraphonomyIntraGraphReasoning( n_in_channels = 3, n_classes = args.n_classes, n_node_features = args.n_node_features, n_output_channels = args.n_output_channels ).to(device)
    model_D = PatchGANDiscriminator( n_in_channels = args.n_output_channels, n_fmaps = 64 ).to( device )
    if( args.debug ):
        print( "model_G\n", model_G )

    # モデルを読み込む
    if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
        load_checkpoint(model_G, device, args.load_checkpoints_path )
        
    #================================
    # optimizer_G の設定
    #================================
    optimizer_G = optim.Adam( params = model_G.parameters(), lr = args.lr, betas = (args.beta1,args.beta2) )
    optimizer_D = optim.Adam( params = model_D.parameters(), lr = args.lr, betas = (args.beta1,args.beta2) )

    #================================
    # loss 関数の設定
    #================================
    if( args.n_output_channels == 1 ):
        loss_l1_fn = nn.L1Loss()
        loss_vgg_fn = VGGLoss(device, n_channels = args.n_output_channels)
        loss_adv_fn = LSGANLoss(device)
    else:
        loss_bce_fn = ParsingCrossEntropyLoss()
        #loss_bce_fn = CrossEntropy2DLoss(device)

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
            model_G.train()
            model_D.train()

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

            #----------------------------------------------------
            # 生成器 の forword 処理
            #----------------------------------------------------
            output, embedded, graph, reproj_feature = model_G( image, adj_matrix_cihp_to_cihp )
            _, output_vis = torch.max(output, 1)
            output_vis_rgb = decode_labels_tsr(output_vis)

            if( args.debug and n_print > 0 ):
                print( "output.shape : ", output.shape )
                print( "output_vis.shape : ", output_vis.shape )
                print( "output_vis_rgb.shape : ", output_vis_rgb.shape )
                print( "embedded.shape : ", embedded.shape )
                print( "graph.shape : ", graph.shape )
                print( "reproj_feature.shape : ", reproj_feature.shape )

            #----------------------------------------------------
            # 識別器の更新処理
            #----------------------------------------------------
            # 無効化していた識別器 D のネットワークの勾配計算を有効化。
            for param in model_D.parameters():
                param.requires_grad = True

            # 学習用データをモデルに流し込む
            d_real = model_D( output )
            d_fake = model_D( output.detach() )
            if( args.debug and n_print > 0 ):
                print( "d_real.shape :", d_real.shape )
                print( "d_fake.shape :", d_fake.shape )

            # 損失関数を計算する
            loss_D, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )

            # ネットワークの更新処理
            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # 無効化していた識別器 D のネットワークの勾配計算を有効化。
            for param in model_D.parameters():
                param.requires_grad = False

            #----------------------------------------------------
            # 生成器の更新処理
            #----------------------------------------------------
            # 損失関数を計算する
            if( args.n_output_channels == 1 ):
                loss_l1 = loss_l1_fn( output, target )
                loss_vgg = loss_vgg_fn( output, target )
                loss_adv = loss_adv_fn.forward_G( d_fake )
                loss_G = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_adv * loss_adv
            else:
                loss_G = loss_bce_fn( output, target )

            # ネットワークの更新処理
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                # loss
                if( args.n_output_channels == 1 ):
                    board_train.add_scalar('G/loss_G', loss_G.item(), step)
                    board_train.add_scalar('G/loss_l1', loss_l1.item(), step)
                    board_train.add_scalar('G/loss_vgg', loss_vgg.item(), step)
                    board_train.add_scalar('G/loss_adv', loss_adv.item(), step)
                    board_train.add_scalar('D/loss_D', loss_D.item(), step)
                    board_train.add_scalar('D/loss_D_real', loss_D_real.item(), step)
                    board_train.add_scalar('D/loss_D_fake', loss_D_fake.item(), step)
                    print( "step={}, loss_G={:.5f}, loss_l1={:.5f}, loss_vgg={:.5f}, loss_adv={:.5f}".format(step, loss_G.item(), loss_l1.item(), loss_vgg.item(), loss_adv.item()) )
                    print( "step={}, loss_D={:.5f}, loss_D_real={:.5f}, loss_D_fake={:.5f}".format(step, loss_D.item(), loss_D_real.item(), loss_D_fake.item()) )
                else:
                    board_train.add_scalar('G/loss', loss.item(), step)
                    print( "step={}, loss={:.5f}".format(step, loss.item()) )

                # visual images
                if( args.n_output_channels == 1 ):
                    visuals = [
                        [ image, target, output ],
                    ]
                else:
                    visuals = [
                        [ image, target, output_vis.unsqueeze(1), output_vis_rgb ],
                        [ output[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes_source//2) ],
                        [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes_source//2 + 1,args.n_classes_source) ],
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

            #====================================================
            # valid データでの処理
            #====================================================
            if( step == 0 or ( step % args.n_display_valid_step == 0 ) ):
                loss_G_total, loss_l1_total, loss_vgg_total, loss_adv_total = 0, 0, 0, 0
                loss_D_total, loss_D_real_total, loss_D_fake_total = 0, 0, 0
                n_valid_loop = 0
                for iter, inputs in enumerate( tqdm(dloader_valid, desc = "eval iters") ):
                    model_G.eval()            
                    model_D.eval()

                    # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                    if inputs["image"].shape[0] != args.batch_size_valid:
                        break

                    # ミニバッチデータを GPU へ転送
                    image = inputs["image"].to(device)
                    target = inputs["target"].to(device)

                    # 推論処理
                    with torch.no_grad():
                        output, embedded, graph, reproj_feature = model_G( image, adj_matrix_cihp_to_cihp )
                        _, output_vis = torch.max(output, 1)
                        output_vis_rgb = decode_labels_tsr(output_vis)

                    with torch.no_grad():
                        d_real = model_D( output )
                        d_fake = model_D( output.detach() )

                    # 損失関数を計算する
                    if( args.n_output_channels == 1 ):
                        # 生成器
                        loss_l1 = loss_l1_fn( output, target )
                        loss_vgg = loss_vgg_fn( output, target )
                        loss_adv = loss_adv_fn.forward_G( d_fake )
                        loss_G = args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_adv * loss_adv
                        loss_G_total += loss_G
                        loss_l1_total += loss_l1
                        loss_vgg_total += loss_vgg
                        loss_adv_total += loss_adv

                        # 識別器
                        loss_D, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )
                        loss_D_total += loss_D
                        loss_D_real_total += loss_D_real
                        loss_D_fake_total += loss_D_fake
                    else:
                        loss = loss_bce_fn( output, target )
                        loss_G_total += loss

                    # 生成画像表示
                    if( iter <= args.n_display_valid ):
                        # visual images
                        if( args.n_output_channels == 1 ):
                            visuals = [
                                [ image, target, output ],
                            ]
                        else:
                            visuals = [
                                [ image, target, output_vis.unsqueeze(1), output_vis_rgb ],
                                [ output[:,i,:,:].unsqueeze(1) for i in range(0,args.n_classes_source//2) ],
                                [ output[:,i,:,:].unsqueeze(1) for i in range(args.n_classes_source//2 + 1,args.n_classes_source) ],
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
                if( args.n_output_channels == 1 ):
                    board_valid.add_scalar('G/loss_G', loss_G_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_l1', loss_l1_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_vgg', loss_vgg_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_adv', loss_adv_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('D/loss_D', loss_D_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('D/loss_D_real', loss_D_real_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('D/loss_D_fake', loss_D_fake_total.item()/n_valid_loop, step)
                else:
                    board_valid.add_scalar('G/loss', loss_G_total.item()/n_valid_loop, step)
                
            step += 1
            n_print -= 1

        #====================================================
        # モデルの保存
        #====================================================
        if( epoch % args.n_save_epoches == 0 ):
            save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_ep%03d.pth' % (epoch)) )
            save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth') )
            print( "saved checkpoints" )

    print("Finished Training Loop.")
    save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_final.pth') )
