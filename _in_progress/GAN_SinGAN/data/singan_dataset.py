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
import torch.nn as nn
from torchvision.utils import save_image

# 自作モジュール
from data.transforms.random_erasing import RandomErasing
from data.transforms.tps_transform import TPSTransform
from utils import set_random_seed, numerical_sort

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class SinGANDataset(data.Dataset):
    def __init__(
        self, 
        args, dataset_dir, datamode = "train", image_height = 128, image_width = 128, 
        train_progress_init = 0, train_progress_max = 8, scale_factor = 0.77823, 
        n_layers=5, kernel_size=3,     
        data_augument = False, debug = False,
    ):
        super(SinGANDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.datamode = datamode
        self.train_progress_max = train_progress_max
        self.data_augument = data_augument
        self.image_height = image_height
        self.image_width = image_width

        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.debug = debug

        self.noize_sizes_h = [self.image_height]
        self.noize_sizes_w = [self.image_width]
        for i in range(train_progress_max):
            self.noize_sizes_h.insert(0, int(self.noize_sizes_h[0] * scale_factor) )
            self.noize_sizes_w.insert(0, int(self.noize_sizes_w[0] * scale_factor) )

        #self.noize_sizes_h = [25,33,42,54,69,88,113,145,186]
        #self.noize_sizes_w = [34,43,56,71,91,117,151,193,248]

        self.image_dir = os.path.join( self.dataset_dir )
        self.image_names = sorted( [f for f in os.listdir(self.image_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )

        if( self.debug ):
            print( "self.image_dir :", self.image_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])
            print( "self.noize_sizes_h : ", self.noize_sizes_h )
            print( "self.noize_sizes_w : ", self.noize_sizes_w )

        return

    def __len__(self):
        return len(self.image_names)

    def get_gan_noize_image_z(self, c, h, w, padding, noize_type = "uniform" ):
        if( noize_type == "uniform" ):
            noize_z = torch.randn( size = (1,h,w) )
        else:
            NotImplementedError()

        # チャネル次元は同じ値にする
        noize_z = noize_z.expand(c,h,w)

        # zero padding で境界をゼロ埋め
        zero_pad = nn.ZeroPad2d(padding)
        noize_z = zero_pad(noize_z)
        return noize_z

    def __getitem__(self, index):
        image_name = self.image_names[index]
        self.seed_da = random.randint(0,10000)

        #--------------------------------
        # 入力ノイズ画像
        #--------------------------------
        noize_image_z_list = []
        for train_progress in range(self.train_progress_max):
            noize_h = self.noize_sizes_h[train_progress]
            noize_w = self.noize_sizes_w[train_progress]
            noise_padding = int(((self.kernel_size - 1) * self.n_layers) / 2)
            #print( "noise_padding : ", noise_padding )

            # input noize
            noize_image_z = self.get_gan_noize_image_z( c=3, h=noize_h, w=noize_w, padding=noise_padding, noize_type="uniform" )
            #print( "noize_image_z.shape : ", noize_image_z.shape )
            noize_image_z_list.append(noize_image_z)

        #--------------------------------
        # 正解画像
        #--------------------------------
        if( self.datamode in ["train", "valid"] ):
            image_gt_list = []
            for train_progress in range(self.train_progress_max):
                noize_h = self.noize_sizes_h[train_progress]
                noize_w = self.noize_sizes_w[train_progress]

                # transform
                self.transform = transforms.Compose(
                    [
                        transforms.Resize( (noize_h, noize_w), interpolation=Image.LANCZOS ),
                        transforms.CenterCrop( size = (noize_h, noize_w) ),
                        transforms.ToTensor(),
                        transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                    ]
                )

                # image
                image_gt = Image.open( os.path.join(self.image_dir,image_name) ).convert('RGB')
                image_gt = self.transform(image_gt)
                image_gt_list.append(image_gt)

        if( self.datamode in ["train", "valid"] ):
            results_dict = {
                "noize_image_z_list" : noize_image_z_list,
                "image_name" : image_name,
                "image_gt_list" : image_gt_list,
            }
        else:
            results_dict = {
                "noize_image_z_list" : noize_image_z_list,
            }

        return results_dict


class SinGANDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(SinGANDataLoader, self).__init__()
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