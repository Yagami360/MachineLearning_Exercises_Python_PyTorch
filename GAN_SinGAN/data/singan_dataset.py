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
        args, dataset_dir, datamode = "train", train_progress = 1, image_height = 128, image_width = 128, 
        scale_factor = 0.75, scale_factor_stop = 8,
        n_fmaps=32, n_layers=5, kernel_size=3, stride=1, padding=0,
        data_augument = False, debug = False,
    ):
        super(SinGANDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.datamode = datamode
        self.train_progress = train_progress
        self.data_augument = data_augument
        self.image_height = image_height
        self.image_width = image_width

        self.scale_factor = scale_factor
        self.scale_factor_stop = scale_factor_stop

        self.n_fmaps = n_fmaps
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.debug = debug

        self.image_dir = os.path.join( self.dataset_dir )
        self.image_names = sorted( [f for f in os.listdir(self.image_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )

        # transform
        self.transform = transforms.Compose(
            [
                transforms.Resize( (self.image_height, self.image_width), interpolation=Image.LANCZOS ),
                transforms.CenterCrop( size = (self.image_height, self.image_width) ),
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

        scale = math.pow( self.scale_factor, self.scale_factor_stop - self.train_progress )
        print( "scale : ", scale )
        noize_h = 25
        noize_w = 34
        noise_padding = 5
        noize_h = ( self.kernel_size - 1 ) * self.n_layers
        noize_w = ( self.kernel_size - 1 ) * self.n_layers
        noise_padding = int(((self.kernel_size - 1) * self.n_layers) / 2)
        print( "noize_h : ", noize_h )
        print( "noize_w : ", noize_w )
        self.image_height = 35
        self.image_width = 44

        # input noize
        if( self.train_progress == 1 ):
            noize_image_z = self.get_gan_noize_image_z( c=3, h=noize_h, w=noize_w, padding=noise_padding, noize_type="uniform" )
        else:
            NotImplementedError()

        # image
        if( self.datamode in ["train", "valid"] ):
            image_gt = Image.open( os.path.join(self.image_dir,image_name) ).convert('RGB')
            image_gt = self.transform(image_gt)

        if( self.datamode in ["train", "valid"] ):
            results_dict = {
                "noize_image_z" : noize_image_z,
                "image_name" : image_name,
                "image_gt" : image_gt,
            }
        else:
            results_dict = {
                "noize_z" : noize_z,
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