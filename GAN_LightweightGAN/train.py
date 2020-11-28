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
from data.noize_dataset import NoizeDataset
from models.generators import LightweightGANGenerator
from models.discriminators import LightweightGANDiscriminator
from models.inception import InceptionV3
from models.losses import VGGLoss, LSGANLoss
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm
from utils.scores import calculate_fretchet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="dataset/templete_dataset")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=100, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=4, help="バッチサイズ")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_size', type=int, choices=[128, 256, 512, 1024], default=1024, help="出力画像の解像度")
    parser.add_argument('--z_dims', type=int, default=256, help="入力ノイズの次元数")
    parser.add_argument('--lr', type=float, default=0.0002, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--lambda_l1', type=float, default=10.0, help="L1損失関数の係数値")
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help="VGG perceptual loss の係数値")
    parser.add_argument('--lambda_adv', type=float, default=1.0, help="Adv loss の係数値")
    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=10,)
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument('--diaplay_scores', action='store_true')
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
    ds_train = NoizeDataset( args, args.dataset_dir, datamode = "train", image_size = args.image_size, z_dims = args.z_dims, debug = args.debug )

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
    model_G = LightweightGANGenerator( z_dims = args.z_dims, n_fmaps = 64, out_dims = 3, image_size = args.image_size ).to(device)
    model_D = LightweightGANDiscriminator( in_dim = 3, n_fmaps = 64, image_size = args.image_size ).to( device )

    # モデルを読み込む
    if not args.load_checkpoints_path == '' and os.path.exists(args.load_checkpoints_path):
        load_checkpoint(model_G, device, args.load_checkpoints_path )

    if( args.debug ):
        print( "model_G\n", model_G )
        print( "model_D\n", model_D )

    # Inception モデル / FID スコアの計算用
    if( args.diaplay_scores ):
        inception = InceptionV3().to(device)

    #================================
    # optimizer_G の設定
    #================================
    optimizer_G = optim.Adam( params = model_G.parameters(), lr = args.lr, betas = (args.beta1,args.beta2) )
    optimizer_D = optim.Adam( params = model_D.parameters(), lr = args.lr, betas = (args.beta1,args.beta2) )

    #================================
    # loss 関数の設定
    #================================
    loss_l1_fn = nn.L1Loss()
    loss_vgg_fn = VGGLoss(device, n_channels=3)
    loss_adv_fn = LSGANLoss(device)

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    n_print = 1
    step = 0
    for epoch in tqdm( range(args.n_epoches), desc = "epoches" ):
        for iter, inputs in enumerate( tqdm( dloader_train, desc = "epoch={}".format(epoch) ) ):
            model_G.train()
            model_D.train()

            # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
            if inputs["image_t"].shape[0] != args.batch_size:
                break

            # ミニバッチデータを GPU へ転送
            latent_z = inputs["latent_z"].to(device)
            image_t = inputs["image_t"].to(device)
            if( args.debug and n_print > 0):
                print( "[latent_z] shape={}, dtype={}, min={}, max={}".format(latent_z.shape, latent_z.dtype, torch.min(latent_z), torch.max(latent_z)) )
                print( "[image_t] shape={}, dtype={}, min={}, max={}".format(image_t.shape, image_t.dtype, torch.min(image_t), torch.max(image_t)) )

            #----------------------------------------------------
            # 生成器 の forword 処理
            #----------------------------------------------------
            output, output_res128 = model_G( latent_z )
            if( args.debug and n_print > 0 ):
                print( "output.shape : ", output.shape )
                print( "output_res128.shape : ", output_res128.shape )

            #----------------------------------------------------
            # 識別器の更新処理
            #----------------------------------------------------
            # 無効化していた識別器 D のネットワークの勾配計算を有効化。
            for param in model_D.parameters():
                param.requires_grad = True

            # 学習用データをモデルに流し込む
            d_outputs_real = model_D( image_t )
            d_outputs_fake = model_D( output.detach(), output_res128.detach() )
            d_real = d_outputs_real["d_output"]
            d_fake = d_outputs_fake["d_output"]
            if( args.debug and n_print > 0 ):
                print( "d_real.shape :", d_real.shape )
                print( "d_fake.shape :", d_fake.shape )
                for key, value in d_outputs_real.items():
                    print('[d_outputs_real] {} : shape={}, dype={}'.format(str(key), value.shape, value.dtype) )
                for key, value in d_outputs_fake.items():
                    print('[d_outputs_fake] {} : shape={}, dype={}'.format(str(key), value.shape, value.dtype) )

            # 損失関数を計算する
            _, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )
            loss_D_rec_f2 = loss_vgg_fn( d_outputs_real["rec_img_res128"], d_outputs_real["rec_img_f2"] )
            loss_D = loss_D_real + loss_D_fake + loss_D_rec_f2

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
            loss_l1 = loss_l1_fn( image_t, output )
            loss_vgg = loss_vgg_fn( image_t, output )
            loss_adv = loss_adv_fn.forward_G( d_fake )
            loss_G =  args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_adv * loss_adv

            # ネットワークの更新処理
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            #====================================================
            # 学習過程の表示
            #====================================================
            if( step == 0 or ( step % args.n_diaplay_step == 0 ) ):
                # lr
                for param_group in optimizer_G.param_groups:
                    lr = param_group['lr']

                board_train.add_scalar('lr/learning rate', lr, step )

                # loss
                board_train.add_scalar('G/loss_l1', loss_l1.item(), step)
                board_train.add_scalar('G/loss_vgg', loss_vgg.item(), step)
                board_train.add_scalar('G/loss_adv', loss_adv.item(), step)
                board_train.add_scalar('G/loss_G', loss_G.item(), step)
                board_train.add_scalar('D/loss_D_real', loss_D_real.item(), step)
                board_train.add_scalar('D/loss_D_fake', loss_D_fake.item(), step)
                board_train.add_scalar('D/loss_D_rec_f2', loss_D_rec_f2.item(), step)
                board_train.add_scalar('D/loss_D', loss_D.item(), step)
                print( "step={}, loss_G={:.5f}, loss_l1={:.5f}, loss_vgg={:.5f}, loss_adv={:.5f}".format(step, loss_G.item(), loss_l1.item(), loss_vgg.item(), loss_adv.item()) )
                print( "step={}, loss_D={:.5f}, loss_D_real={:.5f}, loss_D_fake={:.5f}, loss_D_rec_f2={:.5f}".format(step, loss_D.item(), loss_D_real.item(), loss_D_fake.item(), loss_D_rec_f2.item()) )

                # visual images
                visuals = [
                    [ image_t.detach(), output.detach() ],
                ]
                board_add_images(board_train, 'train', visuals, step+1)

                # scores
                if( args.diaplay_scores ):
                    score_fid = calculate_fretchet(image_t, output, inception)
                    board_train.add_scalar('scores/FID', score_fid.item(), step)
                    print( "step={}, FID={:.5f}".format(step, score_fid.item()) )

            #====================================================
            # valid データでの処理
            #====================================================
            if( step % args.n_display_valid_step == 0 ):
                loss_G_total, loss_l1_total, loss_vgg_total, loss_adv_total = 0, 0, 0, 0
                loss_D_total, loss_D_real_total, loss_D_fake_total, loss_D_rec_f2_total = 0, 0, 0, 0
                score_fid_total = 0
                n_valid_loop = 0
                for iter, inputs in enumerate( tqdm(dloader_valid, desc = "valid") ):
                    model_G.eval()            
                    model_D.eval()

                    # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                    if inputs["image_t"].shape[0] != args.batch_size_valid:
                        break

                    # ミニバッチデータを GPU へ転送
                    latent_z = inputs["latent_z"].to(device)
                    image_t = inputs["image_t"].to(device)

                    # 推論処理
                    with torch.no_grad():
                        output, output_res128 = model_G( latent_z )

                    with torch.no_grad():
                        d_outputs_real = model_D( image_t )
                        d_outputs_fake = model_D( output.detach(), output_res128.detach() )
                        d_real = d_outputs_real["d_output"]
                        d_fake = d_outputs_fake["d_output"]

                    # 損失関数を計算する
                    loss_l1 = loss_l1_fn( image_t, output )
                    loss_vgg = loss_vgg_fn( image_t, output )
                    loss_adv = loss_adv_fn.forward_G( d_fake )
                    loss_G =  args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_adv * loss_adv

                    loss_l1_total += loss_l1
                    loss_vgg_total += loss_vgg
                    loss_adv_total += loss_adv
                    loss_G_total += loss_G

                    _, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )
                    loss_D_rec_f2 = loss_vgg_fn( d_outputs_real["rec_img_res128"], d_outputs_real["rec_img_f2"] )
                    loss_D = loss_D_real + loss_D_fake + loss_D_rec_f2
                    loss_D_total += loss_D
                    loss_D_real_total += loss_D_real
                    loss_D_fake_total += loss_D_fake
                    loss_D_rec_f2_total += loss_D_rec_f2

                    # scores
                    if( args.diaplay_scores ):
                        score_fid = calculate_fretchet(image_t, output, inception)
                        score_fid_total += score_fid

                    # 生成画像表示
                    if( iter <= args.n_display_valid ):
                        # visual images
                        visuals = [
                            [ image_t.detach(), output.detach() ],
                        ]
                        board_add_images(board_valid, 'valid/{}'.format(iter), visuals, step+1)

                    n_valid_loop += 1

                # loss 値表示
                board_valid.add_scalar('G/loss_l1', loss_l1_total.item()/n_valid_loop, step)
                board_valid.add_scalar('G/loss_vgg', loss_vgg_total.item()/n_valid_loop, step)
                board_valid.add_scalar('G/loss_adv', loss_adv_total.item()/n_valid_loop, step)
                board_valid.add_scalar('G/loss_G', loss_G_total.item()/n_valid_loop, step)
                board_valid.add_scalar('D/loss_D', loss_D_total.item()/n_valid_loop, step)
                board_valid.add_scalar('D/loss_D_real', loss_D_real_total.item()/n_valid_loop, step)
                board_valid.add_scalar('D/loss_D_fake', loss_D_fake_total.item()/n_valid_loop, step)
                board_valid.add_scalar('D/loss_D_rec_f2', loss_D_rec_f2_total.item()/n_valid_loop, step)

                # scores
                if( args.diaplay_scores ):
                    board_valid.add_scalar('scores/FID', score_fid_total.item()/n_valid_loop, step)

            step += 1
            n_print -= 1

        #====================================================
        # モデルの保存
        #====================================================
        if( epoch % args.n_save_epoches == 0 ):
            save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_G_ep%03d.pth' % (epoch)) )
            save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_G_final.pth') )
            save_checkpoint( model_D, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_D_ep%03d.pth' % (epoch)) )
            save_checkpoint( model_D, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_D_final.pth') )
            print( "saved checkpoints" )

    print("Finished Training Loop.")
    save_checkpoint( model_G, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_G_final.pth') )
    save_checkpoint( model_D, device, os.path.join(args.save_checkpoints_dir, args.exper_name, 'model_D_final.pth') )
