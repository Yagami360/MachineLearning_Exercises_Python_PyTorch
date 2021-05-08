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
from torchvision.utils import save_image

from data.transforms.random_erasing import RandomErasing
from data.transforms.tps_transform import TPSTransform
from utils import set_random_seed, numerical_sort

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class DogsVSCatsDataset(data.Dataset):
    def __init__(self, args, root_dir, datamode = "train", image_height = 128, image_width = 128, data_augument = False, debug = False ):
        super(DogsVSCatsDataset, self).__init__()
        self.args = args
        self.root_dir = root_dir
        self.datamode = datamode
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.data_augument = data_augument
        self.debug = debug

        self.dataset_dir = os.path.join( root_dir, datamode )
        self.image_names = sorted( [f for f in os.listdir(self.dataset_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )

        # データをロードした後に行う各種前処理の関数を構成を指定する。
        #mean = (0.485, 0.456, 0.406)
        #std = (0.229, 0.224, 0.225)
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        if( self.data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (self.image_height, self.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
                    transforms.RandomPerspective(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    transforms.CenterCrop( size = (self.image_height, self.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( mean, std ),
                    RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (self.image_height, self.image_width), interpolation=Image.LANCZOS ),
                    transforms.CenterCrop( size = (self.image_height, self.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( mean, std ),
                ]
            )

        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])

        return

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]

        #----------------
        # 入力画像
        #----------------
        image_pillow = Image.open(os.path.join(self.dataset_dir, image_name)).convert('RGB')
        image = self.transform(image_pillow)

        #----------------
        # 正解ラベル
        #----------------
        if( self.datamode == "train" ):
            if( "cat." in image_name ):
                #target = torch.eye(2)[0].long()
                target = torch.zeros(1).squeeze().long()
            elif( "dog." in image_name ):
                #target = torch.eye(2)[1].long()
                target = torch.ones(1).squeeze().long()
            else:
                #target = torch.eye(2)[0].long()
                target = torch.zeros(1).squeeze().long()

        #----------------
        # 戻り値
        #----------------
        if( self.datamode == "train" ):
            results_dict = {
                "image_name" : image_name,
                "image" : image,
                "target" : target,
            }
        else:
            results_dict = {
                "image_name" : image_name,
                "image" : image,
            }

        return results_dict


class DogsVSCatsDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(DogsVSCatsDataLoader, self).__init__()
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
