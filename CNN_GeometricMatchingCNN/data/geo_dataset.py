import os
import numpy as np
import random
import pandas as pd
import re
import math
from PIL import Image, ImageDraw, ImageOps
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# PyTorch
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.utils import save_image

# 自作モジュール
from utils import set_random_seed

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class GeoDataset(data.Dataset):
    """
    幾何学的変換用データセット / 参照画像と（目標画像のための）ランダムに生成した変換パラメータ θ のペアデータ
    """
    def __init__( 
        self, 
        args, dataset_dir, image_height = 240, image_width = 240, data_augument = False, 
        geometric_model = "affine", 
        random_t_tps = 0.4,     # TPS 変換時の θ 生成のためのランダムパラメータ
        debug = False
    ):
        super(GeoDataset, self).__init__()
        self.args = args
        self.data_augument = data_augument
        self.image_height = image_height
        self.image_width = image_width
        self.geometric_model = geometric_model
        self.random_t_tps = random_t_tps
        self.debug = debug
        self.image_dir = dataset_dir
        self.image_names = sorted( [f for f in os.listdir(self.image_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )

        # transform
        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                ]
            )

        if( self.debug ):
            print( "self.image_dir :", self.image_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])

        return

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        #--------------------
        # image_s / 変換前画像
        #--------------------
        image_s = Image.open( os.path.join(self.image_dir,image_name) ).convert('RGB')
        self.seed_da = random.randint(0,10000)
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        image_s = self.transform(image_s)
        #print( "image_s.shape : ", image_s.shape )

        #-----------------------------------------------
        # theta_gt / 変形パラメータ θ の教師データ
        # Synthetic image_s generation により手動のアノテーションを行うことなく自動的に生成する
        #-----------------------------------------------
        if( self.geometric_model == "affine" ):
            #theta_gt = torch.zeros( 6, requires_grad = False ).float()   # dummy
            rot_angle = (np.random.rand(1)-0.5) * 2 * np.pi/12 # between -np.pi/12 and np.pi/12
            sh_angle = (np.random.rand(1)-0.5) * 2 * np.pi/6   # between -np.pi/6 and np.pi/6
            lambda_1 = 1 + (2*np.random.rand(1)-1) * 0.25      # between 0.75 and 1.25
            lambda_2 = 1 + (2*np.random.rand(1)-1) * 0.25      # between 0.75 and 1.25
            tx = (2*np.random.rand(1)-1) * 0.25                # between -0.25 and 0.25
            ty = (2*np.random.rand(1)-1) * 0.25
            R_sh = np.array( [[np.cos(sh_angle[0]),-np.sin(sh_angle[0])], [np.sin(sh_angle[0]),np.cos(sh_angle[0])]] )
            R_alpha = np.array( [[np.cos(rot_angle[0]),-np.sin(rot_angle[0])], [np.sin(rot_angle[0]),np.cos(rot_angle[0])]] )
            D = np.diag([lambda_1[0],lambda_2[0]])
            A = R_alpha @ R_sh.transpose() @ D @ R_sh
            theta_gt = np.array([A[0,0],A[0,1],tx,A[1,0],A[1,1],ty])
            theta_gt = torch.Tensor( theta_gt.astype(np.float32) )
        elif( self.geometric_model == "tps" ):
            #theta_gt = torch.zeros( 18, requires_grad = False ).float()   # dummy
            theta_gt = np.array([-1 , -1 , -1 , 0 , 0 , 0 , 1 , 1 , 1 , -1 , 0 , 1 , -1 , 0 , 1 , -1 , 0 , 1])
            theta_gt = theta_gt + (np.random.rand(18)-0.5) * 2 * self.random_t_tps
            theta_gt = torch.Tensor( theta_gt.astype(np.float32) )
        elif( self.geometric_model == "hom" ):
            #theta_gt = torch.zeros( 9, requires_grad = False ).float()    # dummy
            theta_gt = np.array([-1, -1, 1, 1, -1, 1, -1, 1])
            theta_gt = theta_gt + (np.random.rand(8)-0.5) * 2 * self.random_t_tps
            theta_gt = torch.Tensor( theta_gt.astype(np.float32) )
        else:
            NotImplementedError()
            
        #-----------------------------------------------
        # return 値の設定
        #-----------------------------------------------
        results_dict = {
            "image_name" : image_name,
            "image_s" : image_s,                # 参照画像
            "theta_gt" : theta_gt,              # theta の正解データ
        }
        return results_dict


class GeoDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(GeoDataLoader, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle = shuffle,
                num_workers = n_workers,
                pin_memory = pin_memory,
        )

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch