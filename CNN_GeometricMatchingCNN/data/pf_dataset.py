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

class PFDataset(data.Dataset):
    def __init__( self, args, dataset_dir, image_height = 240, image_width = 240, debug = False ):
        super(PFDataset, self).__init__()
        self.args = args
        self.image_height = image_height
        self.image_width = image_width
        self.debug = debug

        self.image_s_dir = os.path.join( dataset_dir, "car(G)" )
        self.image_t_dir = os.path.join( dataset_dir, "car(M)" )
        self.image_s_names = sorted( [f for f in os.listdir(self.image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )
        self.image_t_names = sorted( [f for f in os.listdir(self.image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )

        # transform
        self.transform = transforms.Compose(
            [
                transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                transforms.ToTensor(),
                transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
            ]
        )
        if( self.debug ):
            print( "self.image_s_dir :", self.image_s_dir)
            print( "self.image_t_dir :", self.image_t_dir)
            print( "len(self.image_s_names) :", len(self.image_s_names))
            print( "self.image_s_names[0:5] :", self.image_s_names[0:5])
            print( "self.image_t_names[0:5] :", self.image_t_names[0:5])

        return

    def __len__(self):
        return len(self.image_s_names)

    def __getitem__(self, index):
        image_s_name = self.image_s_names[index]
        image_t_name = self.image_t_names[index]

        #--------------------
        # image_s / 変換前画像
        #--------------------
        image_s = Image.open( os.path.join(self.image_s_dir,image_s_name) ).convert('RGB')
        image_s = self.transform(image_s)

        #-----------------------------------------------
        # image_t / 変換後画像
        #-----------------------------------------------
        image_t = Image.open( os.path.join(self.image_t_dir,image_t_name) ).convert('RGB')
        image_t = self.transform(image_t)

        #-----------------------------------------------
        # return 値の設定
        #-----------------------------------------------
        results_dict = {
            "image_s_name" : image_s_name,
            "image_t_name" : image_t_name,
            "image_s" : image_s,                # 参照画像
            "image_t" : image_t,                # 目標画像
        }

        return results_dict


class PFDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(PFDataLoader, self).__init__()
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