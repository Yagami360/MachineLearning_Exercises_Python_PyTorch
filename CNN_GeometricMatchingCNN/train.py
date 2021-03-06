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
from data.geo_dataset import GeoDataset, GeoDataLoader
from data.pf_dataset import PFDataset, PFDataLoader
from models.geometric_matching_cnn import GeometricMatchingCNN
from models.geo_transform import AffineTransform, TpsTransform, GeoPadTransform
from models.losses import TransformedGridLoss, VGGLoss
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_train_dir", type=str, default="VOCdevkit/VOC2012/JPEGImages")
    parser.add_argument("--dataset_eval_dir", type=str, default="proposal-flow-willow/PF-dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument('--geometric_model', choices=['affine','tps','hom'], default="affine", help="幾何学的変換モデル")
    parser.add_argument('--train_feature_regression', action='store_false', help="Geometric-matching CNN の FeatureRegression 層のみ学習対象とする")
    parser.add_argument("--n_epoches", type=int, default=20, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=16, help="バッチサイズ")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=240, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=240, help="入力画像の幅（pixel単位）")
    parser.add_argument('--lr', type=float, default=0.001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--lr_scheduler', action='store_false', help='Bool (default True), whether to use a decaying lr_scheduler')
    parser.add_argument('--lr_max_iter', type=int, default=1000, help='Number of steps between lr starting value and 1e-6 / (lr default min) when choosing lr_scheduler')
    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=10,)
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument("--lambda_grid", type=float, default=1.0)
    parser.add_argument("--lambda_l1", type=float, default=0.0)
    parser.add_argument("--lambda_vgg", type=float, default=0.0)
    #parser.add_argument("--lambda_adv", type=float, default=0.0)
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
    board_eval = SummaryWriter( log_dir = os.path.join(args.tensorboard_dir, args.exper_name + "_eval") )

    #================================
    # データセットの読み込み
    #================================    
    # 学習用データセットとテスト用データセットの設定
    ds_train = GeoDataset( args, args.dataset_train_dir, image_height = args.image_height, image_width = args.image_width, data_augument = args.data_augument, geometric_model = args.geometric_model, debug = args.debug )

    index = np.arange(len(ds_train))
    train_index, valid_index = train_test_split( index, test_size=args.val_rate, random_state=args.seed )
    if( args.debug ):
        print( "train_index.shape : ", train_index.shape )
        print( "valid_index.shape : ", valid_index.shape )
        print( "train_index[0:10] : ", train_index[0:10] )
        print( "valid_index[0:10] : ", valid_index[0:10] )

    dloader_train = torch.utils.data.DataLoader(Subset(ds_train, train_index), batch_size=args.batch_size, shuffle=True, num_workers = args.n_workers, pin_memory = True )
    dloader_valid = torch.utils.data.DataLoader(Subset(ds_train, valid_index), batch_size=args.batch_size_valid, shuffle=False, num_workers = args.n_workers, pin_memory = True )

    # eval 用データ
    ds_eval = PFDataset( args, args.dataset_eval_dir, image_height = args.image_height, image_width = args.image_width, debug = args.debug )
    dloader_eval = torch.utils.data.DataLoader(ds_eval, batch_size=1, shuffle = False, num_workers = args.n_workers, pin_memory = True )

    #================================
    # モデルの構造を定義する。
    #================================
    # GeometricMatchingCNN モデル
    if( args.geometric_model == "affine" ):
        model_G = GeometricMatchingCNN( n_out_channels = 6 ).to(device)
    elif( args.geometric_model == "tps" ):
        model_G = GeometricMatchingCNN( n_out_channels = 18 ).to(device)
    elif( args.geometric_model == "hom" ):
        model_G = GeometricMatchingCNN( n_out_channels = 9 ).to(device)
    else:
        NotImplementedError()

    # Padding 付き GeoPadTransform
    geo_pad_transform = GeoPadTransform( device = device, image_height = args.image_height, image_width = args.image_width, geometric_model = args.geometric_model )

    # 幾何学的変換モデル
    if( args.geometric_model == "affine" ):
        geo_transform = AffineTransform( device = device, image_height = args.image_height, image_width = args.image_width, n_out_channels = 3, padding_mode = "border" )
    elif( args.geometric_model == "tps" ):
        geo_transform = TpsTransform( device = device, image_height = args.image_height, image_width = args.image_width, use_regular_grid = True, grid_size = 3, reg_factor = 0, padding_mode = "border" )
    else:
        NotImplementedError()

    # モデルを読み込む
    if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
        load_checkpoint(model_G, device, args.load_checkpoints_path )
        #load_checkpoint(model_G.feature_regression, device, args.load_checkpoints_path )

    if( args.debug ):
        print( "model_G :\n", model_G )
        print( "geo_pad_transform :\n", geo_pad_transform )

    #================================
    # optimizer_G の設定
    #================================
    if( args.lr_scheduler ):
        if( args.train_feature_regression ):
            optimizer_G = optim.Adam( params = model_G.feature_regression.parameters(), lr = args.lr )
        else:
            optimizer_G = optim.Adam( params = model_G.parameters(), lr = args.lr )

        scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer_G, T_max = args.lr_max_iter, eta_min = 1e-6 )
    else:
        if( args.train_feature_regression ):
            optimizer_G = optim.Adam( params = model_G.feature_regression.parameters(), lr = args.lr, betas = (args.beta1,args.beta2) )
        else:
            optimizer_G = optim.Adam( params = model_G.parameters(), lr = args.lr, betas = (args.beta1,args.beta2) )

    #================================
    # loss 関数の設定
    #================================
    loss_grid_fn = TransformedGridLoss( device = device, geometric_model = args.geometric_model )
    loss_l1_fn = nn.L1Loss()
    loss_vgg_fn = VGGLoss(device = device, layids = [4] )

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    n_print = 1
    step = 0
    for epoch in tqdm( range(args.n_epoches), desc = "epoches" ):
        for iter, inputs in enumerate( tqdm( dloader_train, desc = "epoch={}".format(epoch) ) ):
            model_G.train()

            # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
            if inputs["image_s"].shape[0] != args.batch_size:
                break

            # ミニバッチデータを GPU へ転送
            image_s = inputs["image_s"].to(device)
            theta_gt = inputs["theta_gt"].to(device)
            if( args.debug and n_print > 0):
                print( "image_s.shape : ", image_s.shape )
                print( "theta_gt.shape : ", theta_gt.shape )

            #----------------------------------------------------
            # ランダムに生成した theta_gt から image_t（目標画像）を生成
            #----------------------------------------------------
            image_s_crop, image_t = geo_pad_transform( image_s, theta_gt )
            if( args.debug and n_print > 0):
                print( "image_s_crop.shape : ", image_s_crop.shape )
                print( "image_t.shape : ", image_t.shape )

            #----------------------------------------------------
            # 生成器 の forword 処理
            #----------------------------------------------------
            theta = model_G( image_s_crop, image_t )
            if( args.debug and n_print > 0 ):
                print( "theta.shape : ", theta.shape )

            # 幾何学的変換モデルを用いて変換パラメータで変形
            warp_image, _ = geo_transform( image_s_crop, theta )
            if( args.debug and n_print > 0 ):
                print( "warp_image.shape : ", warp_image.shape )

            #----------------------------------------------------
            # 生成器の更新処理
            #----------------------------------------------------
            # 損失関数を計算する
            loss_grid = loss_grid_fn( theta, theta_gt )         # theta 間での損失関数 / Geometric-matching CNN の論文&公式実装に準拠
            loss_l1 = loss_l1_fn( warp_image, image_t )         # 変形画像間での L1 Loss / Geometric-matching CNN の論文&公式実装から新規に追加
            loss_vgg = loss_vgg_fn( warp_image, image_t )       # 変形画像間での VGG Loss / Geometric-matching CNN の論文&公式実装から新規に追加
            loss_G = args.lambda_grid * loss_grid + args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg

            # ネットワークの更新処理
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            if( args.lr_scheduler ):
                scheduler_G.step()

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                # lr
                if( args.lr_scheduler ):
                    board_train.add_scalar('lr/learning rate', scheduler_G.get_lr()[-1], step )
                else:
                    for param_group in optimizer_G.param_groups:
                        lr = param_group['lr']

                    board_train.add_scalar('lr/learning rate', lr, step )

                # loss
                board_train.add_scalar('G/loss_G', loss_G.item(), step)
                board_train.add_scalar('G/loss_G', loss_G.item(), step)
                board_train.add_scalar('G/loss_grid', loss_grid.item(), step)
                board_train.add_scalar('G/loss_l1', loss_l1.item(), step)
                board_train.add_scalar('G/loss_vgg', loss_vgg.item(), step)
                print( "step={}, loss_G={:.5f}, loss_grid={:.5f}, loss_ls={:.5f}, loss_vgg={:.5f}".format(step, loss_G.item(), loss_grid.item(), loss_l1.item(), loss_vgg.item() ) )

                # visual images
                zero_tsr = torch.zeros( image_s.shape ).to(device)
                visuals = [
                    [ image_s,  image_s_crop, image_t       ],
                    [ zero_tsr, image_s_crop, warp_image    ],
                ]
                board_add_images(board_train, 'train', visuals, step+1)

            #====================================================
            # valid データでの処理
            #====================================================
            if( step % args.n_display_valid_step == 0 ):
                loss_G_total, loss_grid_total, loss_l1_total, loss_vgg_total = 0, 0, 0, 0
                n_valid_loop = 0
                for iter, inputs in enumerate( tqdm(dloader_valid, desc = "valid") ):
                    model_G.eval()            

                    # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                    if inputs["image_s"].shape[0] != args.batch_size_valid:
                        break

                    # ミニバッチデータを GPU へ転送
                    image_s = inputs["image_s"].to(device)
                    theta_gt = inputs["theta_gt"].to(device)

                    # 推論処理
                    with torch.no_grad():
                        image_s_crop, image_t = geo_pad_transform( image_s, theta_gt )
                        theta = model_G( image_s_crop, image_t )
                        warp_image, _ = geo_transform( image_s_crop, theta )

                    # 損失関数を計算する
                    loss_grid = loss_grid_fn( theta, theta_gt )
                    loss_l1 = loss_l1_fn( warp_image, image_t )
                    loss_vgg = loss_vgg_fn( warp_image, image_t )
                    loss_G = args.lambda_grid * loss_grid + args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg

                    loss_G_total += loss_G
                    loss_grid_total += loss_grid
                    loss_l1_total += loss_l1
                    loss_vgg_total += loss_vgg

                    # 生成画像表示
                    if( iter <= args.n_display_valid ):
                        # visual images
                        zero_tsr = torch.zeros( image_s.shape ).to(device)
                        visuals = [
                            [ image_s,  image_s_crop, image_t       ],
                            [ zero_tsr, image_s_crop, warp_image    ],
                        ]
                        board_add_images(board_valid, 'valid/{}'.format(iter), visuals, step+1)

                    n_valid_loop += 1

                # loss 値表示
                board_valid.add_scalar('G/loss_G', loss_G_total.item()/n_valid_loop, step)
                board_valid.add_scalar('G/loss_grid', loss_grid_total.item()/n_valid_loop, step)
                board_valid.add_scalar('G/loss_l1', loss_l1_total.item()/n_valid_loop, step)
                board_valid.add_scalar('G/loss_vgg', loss_vgg_total.item()/n_valid_loop, step)

            #====================================================
            # eval データでの処理
            #====================================================
            if( step % args.n_display_valid_step == 0 ):
                n_eval_loop = 0
                for iter, inputs in enumerate( tqdm(dloader_eval, desc = "eval") ):
                    model_G.eval()            

                    # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                    if inputs["image_s"].shape[0] != 1:
                        break

                    # ミニバッチデータを GPU へ転送
                    image_s_name = inputs["image_s_name"]
                    image_t_name = inputs["image_t_name"]
                    image_s = inputs["image_s"].to(device)
                    image_t = inputs["image_t"].to(device)

                    # 推論処理
                    with torch.no_grad():
                        theta = model_G( image_s, image_t )

                        # 幾何学的変換モデルを用いて変換パラメータで変形
                        warp_image, grid = geo_transform( image_s, theta )

                    # 生成画像表示
                    if( iter <= args.n_display_valid ):
                        visuals = [
                            [ image_s, image_t, warp_image ],
                        ]
                        board_add_images(board_eval, 'eval/{}'.format(iter), visuals, step+1)

            step += 1
            n_print -= 1

        #====================================================
        # モデルの保存
        #====================================================
        if( epoch % args.n_save_epoches == 0 ):
            save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_G_ep%03d.pth' % (epoch)) )
            save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_G_final.pth') )
            print( "saved checkpoints" )

    print("Finished Training Loop.")
    save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_G_final.pth') )
