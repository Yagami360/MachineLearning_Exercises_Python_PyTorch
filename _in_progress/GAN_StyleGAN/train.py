import os
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
import cv2
from math import ceil

try:
    from apex import amp
except ImportError:
    amp = None

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
from models.generators import StyleGANGenerator
from models.discriminators import ProgressiveDiscriminator
from models.inception import InceptionV3
from models.losses import LSGANLoss, HingeGANLoss
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import board_add_image, board_add_images, save_image_w_norm
from utils.scores import calculate_fretchet

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="debug", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="dataset/few-shot-images/100-shot-panda")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--save_checkpoints_dir', type=str, default="checkpoints", help="モデルの保存ディレクトリ")
    parser.add_argument('--load_checkpoints_path', type=str, default="", help="モデルの読み込みファイルのパス")
    parser.add_argument('--tensorboard_dir', type=str, default="tensorboard", help="TensorBoard のディレクトリ")
    parser.add_argument("--n_epoches", type=str, default="4,4,4,4,8,16,32,64,64", help="各解像度スケール（4,8,16,32,64,128,256,512,1024）毎のエポック数のリスト")
    parser.add_argument('--batch_size', type=str, default="256,256,128,64,32,16,8,4,4", help="各解像度スケール（4,8,16,32,64,128,256,512,1024）毎のバッチサイズのリスト")
    parser.add_argument('--batch_size_valid', type=int, default=1, help="バッチサイズ")
    parser.add_argument("--image_size_init", type=int, default=4, help="出力画像の初期解像度")
    parser.add_argument("--image_size_final", type=int, default=1024, help="出力画像の最終解像度")
    parser.add_argument('--z_dims', type=int, default=512, help="入力ノイズの次元数")
    parser.add_argument('--lr', type=str, default="0.0015,0.0015,0.0015,0.0015,0.0015,0.0015,0.002,0.003,0.003", help="各解像度スケール（4,8,16,32,64,128,256,512,1024）毎の学習率のリスト")
    parser.add_argument('--beta1', type=float, default=0.5, help="学習率の減衰率")
    parser.add_argument('--beta2', type=float, default=0.999, help="学習率の減衰率")
    parser.add_argument('--gan_loss_type', choices=['lsgan', 'hinge'], default="hinge", help="Adv loss の種類")
    parser.add_argument('--lambda_l1', type=float, default=0.0, help="L1損失関数の係数値")
    parser.add_argument('--lambda_adv', type=float, default=1.0, help="Adv loss の係数値")
    parser.add_argument("--n_diaplay_step", type=int, default=100,)
    parser.add_argument('--n_display_valid_step', type=int, default=500, help="valid データの tensorboard への表示間隔")
    parser.add_argument("--n_save_epoches", type=int, default=10000,)
    parser.add_argument("--val_rate", type=float, default=0.01)
    parser.add_argument('--n_display_valid', type=int, default=8, help="valid データの tensorboard への表示数")
    parser.add_argument('--diaplay_scores', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="使用デバイス (CPU or GPU)")
    parser.add_argument('--n_workers', type=int, default=4, help="CPUの並列化数（0 で並列化なし）")
    parser.add_argument('--use_cuda_benchmark', action='store_true', help="torch.backends.cudnn.benchmark の使用有効化")
    parser.add_argument('--use_cuda_deterministic', action='store_true', help="再現性確保のために cuDNN に決定論的振る舞い有効化")
    parser.add_argument('--detect_nan', action='store_true')
    parser.add_argument('--use_amp', action='store_true', help="AMP [Automatic Mixed Precision] の使用有効化")
    parser.add_argument('--opt_level', choices=['O0','O1','O2','O3'], default='O1', help='mixed precision calculation mode')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    n_epoches = []
    for epoche in args.n_epoches.split(","):
        n_epoches.append(int(epoche))
    args.n_epoches = n_epoches

    lr = []
    for i in args.lr.split(","):
        lr.append(float(i))
    args.lr = lr

    batch_size = []
    for i in args.batch_size.split(","):
        batch_size.append(int(i))
    args.batch_size = batch_size

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
    ds_train = NoizeDataset( args, args.dataset_dir, datamode = "train", z_dims = args.z_dims, image_size_init = args.image_size_init, image_size_final = args.image_size_final, debug = args.debug )

    # 学習用データセットとテスト用データセットの設定
    index = np.arange(len(ds_train))
    train_index, valid_index = train_test_split( index, test_size=args.val_rate, random_state=args.seed )
    if( args.debug ):
        print( "train_index.shape : ", train_index.shape )
        print( "valid_index.shape : ", valid_index.shape )
        print( "train_index[0:10] : ", train_index[0:10] )
        print( "valid_index[0:10] : ", valid_index[0:10] )

    dloader_train = torch.utils.data.DataLoader(Subset(ds_train, train_index), batch_size=args.batch_size[0], shuffle=True, num_workers = args.n_workers, pin_memory = True )
    dloader_valid = torch.utils.data.DataLoader(Subset(ds_train, valid_index), batch_size=args.batch_size_valid, shuffle=False, num_workers = args.n_workers, pin_memory = True )

    #================================
    # モデルの構造を定義する。
    #================================
    model_G = StyleGANGenerator(in_dim=512, out_dim=3).to(device)
    model_D = ProgressiveDiscriminator().to(device)

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
    optimizer_G = optim.Adam( params = model_G.parameters(), lr = args.lr[0], betas = (args.beta1,args.beta2) )
    optimizer_D = optim.Adam( params = model_D.parameters(), lr = args.lr[0], betas = (args.beta1,args.beta2) )

    #================================
    # apex initialize
    #================================
    if( args.use_amp ):
        [model_D, model_G], [optimizer_D, optimizer_G] = amp.initialize(
            [model_D, model_G], 
            [optimizer_D, optimizer_G], 
            opt_level = args.opt_level,
            num_losses = 2
        )

    #================================
    # loss 関数の設定
    #================================
    loss_l1_fn = nn.L1Loss()
    if( args.gan_loss_type == "lsgan" ):
        loss_adv_fn = LSGANLoss(device)
    elif( args.gan_loss_type == "hinge" ):
        loss_adv_fn = HingeGANLoss()
    else:
        NotImplementedError()

    #================================
    # モデルの学習
    #================================    
    print("Starting Training Loop...")
    progress_init = int(np.log2(args.image_size_init)) - 2
    progress_final = int(np.log2(args.image_size_final)) - 2
    print( f"progress_init={progress_init}, progress_final={progress_final}" )
    n_print = 1
    step = 0
    for progress in range(progress_final):
        # エポック数の更新
        n_epoche = args.n_epoches[progress]

        # バッチサイズの更新
        del dloader_train
        dloader_train = torch.utils.data.DataLoader(Subset(ds_train, train_index), batch_size=args.batch_size[progress], shuffle=True, num_workers = args.n_workers, pin_memory = True )

        # 学習率の更新
        lr = args.lr[progress]
        for pam_group in optimizer_G.param_groups:
            mul = pam_group.get('mul', 1)
            pam_group['lr'] = lr * mul
        for pam_group in optimizer_D.param_groups:
            mul = pam_group.get('mul', 1)
            pam_group['lr'] = lr * mul

        #
        step_per_progress = 0
        step_per_progress_total = n_epoche * len(dloader_train)
        alpha = 0.0

        #--------------------------
        # 各解像度スケールでの学習
        #--------------------------
        for epoch in tqdm( range(n_epoche), desc = "epoches" ):
            for iter, inputs in enumerate( tqdm( dloader_train, desc = "progress={}, alpha={:0.4f}, epoch={}".format(progress, alpha, epoch) ) ):
                model_G.train()
                model_D.train()

                # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                if inputs["latent_z"].shape[0] != args.batch_size:
                    break

                # train progress の更新
                alpha = min( step_per_progress/step_per_progress_total, 1.0 )
                #print( "step_per_progress={}, step_per_progress_total={}, alpha={}".format(step_per_progress, step_per_progress_total, alpha) )

                # ミニバッチデータを GPU へ転送
                latent_z = inputs["latent_z"].to(device)
                noize_map_list = inputs["noize_map_list"]
                for i in range(len(noize_map_list)):
                    noize_map_list[i] = noize_map_list[i].to(device)
                noize_map = noize_map_list[progress]

                image_t_list = inputs["image_t_list"]          
                for i in range(len(image_t_list)):
                    image_t_list[i] = image_t_list[i].to(device)
                image_t = image_t_list[progress].to(device)
                if( args.debug and n_print > 0 ):
                    print( "[latent_z] shape={}, dtype={}, min={}, max={}".format(latent_z.shape, latent_z.dtype, torch.min(latent_z), torch.max(latent_z)) )
                    for i in range(len(noize_map_list)):
                        print( "noize_map_list[{}] shape={}, device={}, dtype={}, min={}, max={}".format(i, noize_map_list[i].shape, noize_map_list[i].device, noize_map_list[i].dtype, torch.min(noize_map_list[i]), torch.max(noize_map_list[i])) )
                    for i in range(len(image_t_list)):
                        print( "image_t_list[{}] shape={}, device={}, dtype={}, min={}, max={}".format(i, image_t_list[i].shape, image_t_list[i].device, image_t_list[i].dtype, torch.min(image_t_list[i]), torch.max(image_t_list[i])) )

                #----------------------------------------------------
                # 生成器 の forword 処理
                #----------------------------------------------------
                output = model_G( latent_z, noize_map_list, progress, alpha )
                if( args.debug and n_print > 0 ):
                    print( "output.shape : ", output.shape )

                #----------------------------------------------------
                # 識別器の更新処理
                #----------------------------------------------------
                """
                # 無効化していた識別器 D のネットワークの勾配計算を有効化。
                for param in model_D.parameters():
                    param.requires_grad = True
                """

                # 学習用データをモデルに流し込む
                d_real = model_D(image_t, progress, alpha )
                d_fake = model_D(output.detach(), progress, alpha )
                if( args.debug and n_print > 0 ):
                    print( "d_real.shape :", d_real.shape )
                    print( "d_fake.shape :", d_fake.shape )

                # 損失関数を計算する
                loss_D, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )

                # ネットワークの更新処理
                optimizer_D.zero_grad()
                if( args.use_amp ):
                    with amp.scale_loss(loss_D, optimizer_D, loss_id=0) as loss_D_scaled:
                        loss_D_scaled.backward()
                else:
                    loss_D.backward()

                optimizer_D.step()

                """
                # 無効化していた識別器 D のネットワークの勾配計算を有効化。
                for param in model_D.parameters():
                    param.requires_grad = False
                """
                #----------------------------------------------------
                # 生成器の更新処理
                #----------------------------------------------------
                d_fake = model_D(output, progress, alpha )

                # 損失関数を計算する
                loss_l1 = loss_l1_fn( image_t, output )
                loss_adv = loss_adv_fn.forward_G( d_fake )
                loss_G =  args.lambda_l1 * loss_l1 + args.lambda_adv * loss_adv

                # ネットワークの更新処理
                optimizer_G.zero_grad()
                if( args.use_amp ):
                    with amp.scale_loss(loss_G, optimizer_G, loss_id=1) as loss_G_scaled:
                        loss_G_scaled.backward()
                else:
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
                    board_train.add_scalar('G/loss_adv', loss_adv.item(), step)
                    board_train.add_scalar('G/loss_G', loss_G.item(), step)
                    board_train.add_scalar('D/loss_D_real', loss_D_real.item(), step)
                    board_train.add_scalar('D/loss_D_fake', loss_D_fake.item(), step)
                    board_train.add_scalar('D/loss_D', loss_D.item(), step)
                    print( "step={}, loss_G={:.5f}, loss_l1={:.5f}, loss_adv={:.5f}".format(step, loss_G.item(), loss_l1.item(), loss_adv.item()) )
                    print( "step={}, loss_D={:.5f}, loss_D_real={:.5f}, loss_D_fake={:.5f}".format(step, loss_D.item(), loss_D_real.item(), loss_D_fake.item()) )

                    # visual images
                    visuals = [
                        [ noize_map.detach(), image_t.detach(), output.detach() ],
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
                    loss_G_total, loss_l1_total, loss_adv_total = 0, 0, 0
                    loss_D_total, loss_D_real_total, loss_D_fake_total = 0, 0, 0
                    score_fid_total = 0
                    n_valid_loop = 0
                    for iter, inputs in enumerate( tqdm(dloader_valid, desc = "valid") ):
                        model_G.eval()            
                        model_D.eval()

                        # 一番最後のミニバッチループで、バッチサイズに満たない場合は無視する（後の計算で、shape の不一致をおこすため）
                        if inputs["latent_z"].shape[0] != args.batch_size_valid:
                            break

                        # ミニバッチデータを GPU へ転送
                        latent_z = inputs["latent_z"].to(device)
                        noize_map_list = inputs["noize_map_list"]
                        for i in range(len(noize_map_list)):
                            noize_map_list[i] = noize_map_list[i].to(device)
                        noize_map = noize_map_list[progress]

                        image_t_list = inputs["image_t_list"]          
                        for i in range(len(image_t_list)):
                            image_t_list[i] = image_t_list[i].to(device)
                        image_t = image_t_list[progress].to(device)

                        # 推論処理
                        with torch.no_grad():
                            output = model_G( latent_z, noize_map_list, progress, alpha )

                        with torch.no_grad():
                            d_real = model_D( image_t, progress, alpha )
                            d_fake = model_D( output.detach(), progress, alpha )
                            print( "[valid] d_real={}, d_fake={}".format(d_real, d_fake) )

                        # 損失関数を計算する
                        #d_fake = model_D(output, progress, alpha )
                        loss_l1 = loss_l1_fn( image_t, output )
                        loss_adv = loss_adv_fn.forward_G( d_fake )
                        loss_G =  args.lambda_l1 * loss_l1 + args.lambda_adv * loss_adv

                        loss_l1_total += loss_l1
                        loss_adv_total += loss_adv
                        loss_G_total += loss_G

                        loss_D, loss_D_real, loss_D_fake = loss_adv_fn.forward_D( d_real, d_fake )
                        loss_D_total += loss_D
                        loss_D_real_total += loss_D_real
                        loss_D_fake_total += loss_D_fake

                        # scores
                        if( args.diaplay_scores ):
                            score_fid = calculate_fretchet(image_t, output, inception)
                            score_fid_total += score_fid

                        # 生成画像表示
                        if( iter <= args.n_display_valid ):
                            # visual images
                            visuals = [
                                [ noize_map.detach(), image_t.detach(), output.detach() ],
                            ]
                            board_add_images(board_valid, 'valid/{}'.format(iter), visuals, step+1)

                        n_valid_loop += 1

                    # loss 値表示
                    board_valid.add_scalar('G/loss_l1', loss_l1_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_adv', loss_adv_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('G/loss_G', loss_G_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('D/loss_D', loss_D_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('D/loss_D_real', loss_D_real_total.item()/n_valid_loop, step)
                    board_valid.add_scalar('D/loss_D_fake', loss_D_fake_total.item()/n_valid_loop, step)
                    
                    # scores
                    if( args.diaplay_scores ):
                        board_valid.add_scalar('scores/FID', score_fid_total.item()/n_valid_loop, step)

                step += 1
                step_per_progress += 1
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
