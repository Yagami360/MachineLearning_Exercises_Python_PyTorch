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

from data.transforms import RandomErasing
from utils import set_random_seed, numerical_sort

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class Neutral2HappinessDataset(data.Dataset):
    def __init__(self, args, dataset_dir, pairs_file = "train_pairs.csv", datamode = "train", image_height = 128, image_width = 128, data_augument = False, debug = False ):
        super(Neutral2HappinessDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.datamode = datamode
        self.data_augument = data_augument
        self.image_height = image_height
        self.image_width = image_width
        self.debug = debug
        self.df_pairs = pd.read_csv( os.path.join(self.dataset_dir, pairs_file) )
        
        # transform
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25, 0.25), scale = (0.80,1.25), resample=Image.BICUBIC ),
                    transforms.RandomPerspective(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                    RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
                ]
            )

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25, 0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.RandomPerspective(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                    RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
                ]
            )

            self.transform_mask_woToTensor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25, 0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.RandomPerspective(),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
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
            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                ]
            )
            self.transform_mask_woToTensor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                ]
            )

        if( self.debug ):
            print( self.df_pairs.head() )

        return

    def __len__(self):
        return len(self.df_pairs)

    def __getitem__(self, index):
        domainA_name = self.df_pairs["domainA_name"].iloc[index]
        domainB_name = self.df_pairs["domainB_name"].iloc[index]
        self.seed_da = random.randint(0,10000)

        # domainA
        domainA = Image.open( os.path.join(self.dataset_dir, "domainA", domainA_name) ).convert('RGB')        
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        domainA = self.transform(domainA)

        # domainB
        if( self.datamode in ["train", "valid"] ):
            domainB_gt = Image.open( os.path.join(self.dataset_dir, "domainB", domainB_name) ).convert('RGB')
            if( self.data_augument ):
                set_random_seed( self.seed_da )

            domainB_gt = self.transform(domainB_gt)

        if( self.datamode in ["train", "valid"] ):
            results_dict = {
                "domainA_name" : domainA_name,
                "domainB_name" : domainB_name,
                "domainA" : domainA,
                "domainB_gt" : domainB_gt,
            }
        else:
            results_dict = {
                "domainA_name" : domainA_name,
                "domainA" : domainA,
            }

        return results_dict


class Neutral2HappinessDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(Neutral2HappinessDataLoader, self).__init__()
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