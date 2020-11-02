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
from data.neutral2happiness_dataset import Neutral2HappinessDataset, Neutral2HappinessDataLoader
from models.generators import Pix2PixHDGenerator, Pix2PixHDAttentionGenerator
from models.discriminators import PatchGANDiscriminator
from models.losses import VGGLoss, LSGANLoss
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm
from utils.decode_labels import decode_labels_tsr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="adain_sample", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="dataset/bu3dfe_neutral2happiness_n12")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_G_path', type=str, default="", help="生成器モデルの読み込みファイルのパス")
    parser.add_argument('--load_checkpoints_D_path', type=str, default="", help="識別器モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=int, default=100, help="エポック数")    
    parser.add_argument('--batch_size', type=int, default=4, help="バッチサイズ")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument('--image_height', type=int, default=128, help="入力画像の高さ（pixel単位）")
    parser.add_argument('--image_width', type=int, default=128, help="入力画像の幅（pixel単位）")
    parser.add_argument('--lr', type=float, default=0.0001, help="学習率")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=100,)
    parser.add_argument("--val_rate", type=float, default=0.30)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--data_augument', action='store_true')
    parser.add_argument('--net_G_type', choices=['pix2pixhd', 'pix2pixhd_attention'], default="pix2pixhd", help="ネットワークの種類")
    parser.add_argument('--lambda_l1', type=float, default=10.0, help="L1損失関数の係数値")
    parser.add_argument('--lambda_vgg', type=float, default=10.0, help="VGG perceptual loss_G の係数値")
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
    ds_train = Neutral2HappinessDataset( args, args.dataset_dir, datamode = "train", image_height = args.image_height, image_width = args.image_width, data_augument = args.data_augument, debug = args.debug )

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
    if( args.net_G_type == "pix2pixhd" ):
        model_G = Pix2PixHDGenerator(input_nc = 3, output_nc = 3).to(device)
    elif( args.net_G_type == "pix2pixhd_attention" ):
        model_G = Pix2PixHDAttentionGenerator(input_nc = 3, output_nc = 3).to(device)
    else:
        NotImplementedError()

    model_D = PatchGANDiscriminator( n_in_channels = 3+3, n_fmaps = 64 ).to( device )
    if( args.debug ):
        print( "model_G\n", model_G )
        print( "model_D\n", model_D )

    # モデルを読み込む
    if not args.load_checkpoints_G_path == '' and os.path.exists(args.load_checkpoints_G_path):
        load_checkpoint(model_G, device, args.load_checkpoints_G_path )
    if not args.load_checkpoints_D_path == '' and os.path.exists(args.load_checkpoints_D_path):
        load_checkpoint(model_D, device, args.load_checkpoints_D_path )

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
            if inputs["domainA"].shape[0] != args.batch_size:
                break

            # ミニバッチデータを GPU へ転送
            domainA = inputs["domainA"].to(device)
            domainB_gt = inputs["domainB_gt"].to(device)

            if( args.debug and n_print > 0):
                print( "[domainA] shape={}, dtype={}, min={}, max={} : ".format(domainA.shape, domainA.dtype, torch.min(domainA), torch.max(domainA) ) )
                print( "[domainB_gt] shape={}, dtype={}, min={}, max={} : ".format(domainB_gt.shape, domainB_gt.dtype, torch.min(domainB_gt), torch.max(domainB_gt) ) )

            #----------------------------------------------------
            # 生成器 の forword 処理
            #----------------------------------------------------
            if( args.net_G_type == "pix2pixhd" ):
                output = model_G( domainA )
            elif( args.net_G_type == "pix2pixhd_attention" ):
                outputs = model_G( domainA )
                output = outputs["output"]
            else:
                NotImplementedError()

            if( args.debug and n_print > 0 ):
                print( "output.shape : ", output.shape )
                if( args.net_G_type == "pix2pixhd_attention" ):
                    print( "outputs[mask].shape : ", outputs["mask"].shape )
                    print( "outputs[content].shape : ", outputs["content"].shape )

            #----------------------------------------------------
            # 識別器の更新処理
            #----------------------------------------------------
            # 無効化していた識別器 D のネットワークの勾配計算を有効化。
            for param in model_D.parameters():
                param.requires_grad = True

            # 学習用データをモデルに流し込む
            d_real = model_D( torch.cat([domainA, domainB_gt], dim=1) )
            d_fake = model_D( torch.cat([domainA, output.detach()], dim=1) )
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
            loss_l1 = loss_l1_fn( domainB_gt, output )
            loss_vgg = loss_vgg_fn( domainB_gt, output )
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
                board_train.add_scalar('D/loss_D', loss_D.item(), step)
                print( "step={}, loss_G={:.5f}, loss_l1={:.5f}, loss_vgg={:.5f}".format(step, loss_G.item(), loss_l1.item(), loss_vgg.item(), loss_adv.item()) )
                print( "step={}, loss_D={:.5f}, loss_D_real={:.5f}, loss_D_fake={:.5f}".format(step, loss_D.item(), loss_D_real.item(), loss_D_fake.item()) )
                
                # visual images
                if( args.net_G_type == "pix2pixhd" ):
                    visuals = [
                        [ domainA, domainB_gt, output ],
                    ]
                elif( args.net_G_type == "pix2pixhd_attention" ):
                    visuals = [
                        [ domainA, domainB_gt, output, outputs["mask"], outputs["content"] ],
                    ]
                else:
                    NotImplementedError()

                board_add_images(board_train, 'train', visuals, step+1)

            #====================================================
            # valid データでの処理
            #====================================================
            if( step % args.n_display_valid_step == 0 ):
                loss_G_total, loss_l1_total, loss_vgg_total, loss_adv_total = 0, 0, 0, 0
                loss_D_total, loss_D_real_total, loss_D_fake_total = 0, 0, 0
                n_valid_loop = 0
                for iter, inputs in enumerate( tqdm(dloader_valid, desc = "valid") ):
                    model_G.eval()
                    model_D.eval()            

                    # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                    if inputs["domainA"].shape[0] != args.batch_size_valid:
                        break

                    # ミニバッチデータを GPU へ転送
                    domainA = inputs["domainA"].to(device)
                    domainB_gt = inputs["domainB_gt"].to(device)

                    # 推論処理
                    with torch.no_grad():
                        if( args.net_G_type == "pix2pixhd" ):
                            output = model_G( domainA )
                        elif( args.net_G_type == "pix2pixhd_attention" ):
                            outputs = model_G( domainA )
                            output = outputs["output"]
                        else:
                            NotImplementedError()

                    with torch.no_grad():
                        d_real = model_D( torch.cat([domainA, domainB_gt], dim=1) )
                        d_fake = model_D( torch.cat([domainA, output.detach()], dim=1) )

                    # 損失関数を計算する
                    loss_l1 = loss_l1_fn( domainB_gt, output )
                    loss_vgg = loss_vgg_fn( domainB_gt, output )
                    loss_adv = loss_adv_fn.forward_G( d_fake )
                    loss_G =  args.lambda_l1 * loss_l1 + args.lambda_vgg * loss_vgg + args.lambda_adv * loss_adv

                    loss_l1_total += loss_l1
                    loss_vgg_total += loss_vgg
                    loss_adv_total += loss_adv
                    loss_G_total += loss_G

                    loss_D, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )
                    loss_D_total += loss_D
                    loss_D_real_total += loss_D_real
                    loss_D_fake_total += loss_D_fake

                    # 生成画像表示
                    if( iter <= args.n_display_valid ):
                        # visual images
                        if( args.net_G_type == "pix2pixhd" ):
                            visuals = [
                                [ domainA, domainB_gt, output ],
                            ]
                        elif( args.net_G_type == "pix2pixhd_attention" ):
                            visuals = [
                                [ domainA, domainB_gt, output, outputs["mask"], outputs["content"] ],
                            ]
                        else:
                            NotImplementedError()

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
