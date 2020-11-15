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
from utils import set_random_seed, onehot_encode_tsr

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class ZalandoDataset(data.Dataset):
    def __init__(self, args, dataset_dir, pairs_file = "train_pairs.csv", datamode = "train", image_height = 128, image_width = 128, n_classes = 20, data_augument = False, onehot = False, debug = False ):
        super(ZalandoDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.pairs_file = pairs_file
        self.datamode = datamode
        self.data_augument= data_augument
        self.onehot = onehot
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes
        self.debug = debug
        self.df_pairs = pd.read_csv( os.path.join(dataset_dir, pairs_file) )

        # transform
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    TPSTransform( tps_points_per_dim = self.args.tps_points_per_dim ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                    #RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
                ]
            )

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    TPSTransform( tps_points_per_dim = self.args.tps_points_per_dim ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                    #RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
                ]
            )

            self.transform_mask_rgb = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    TPSTransform( tps_points_per_dim = self.args.tps_points_per_dim ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                    #RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
                ]
            )

            self.transform_mask_woToTensor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    TPSTransform( tps_points_per_dim = self.args.tps_points_per_dim ),
                    #RandomErasing( probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.5, 0.5, 0.5] ),
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
            self.transform_mask_rgb = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
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
        pose_name = self.df_pairs["pose_name"].iloc[index]
        pose_parsing_name = self.df_pairs["pose_parsing_name"].iloc[index]
        self.seed_da = random.randint(0,10000)

        # pose
        if( self.datamode in ["train", "valid"] ):
            pose_gt = Image.open( os.path.join(self.dataset_dir, "pose", pose_name) ).convert('RGB')
            if( self.data_augument ):
                set_random_seed( self.seed_da )

            pose_gt = self.transform(pose_gt)

        # pose paring
        #pose_parsing_pillow = Image.open( os.path.join(self.dataset_dir, "pose_parsing", pose_parsing_name) ).convert('L')
        pose_parsing_pillow = Image.open( os.path.join(self.dataset_dir, "pose_parsing_rgb", pose_parsing_name) ).convert('RGB')
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        pose_parse = self.transform_mask_rgb(pose_parsing_pillow)
        """
        if( self.onehot ):
            pose_parse = torch.from_numpy( np.asarray(self.transform_mask_woToTensor(pose_parsing_pillow)).astype("int64") ).unsqueeze(0)
            pose_parse = onehot_encode_tsr( pose_parse, n_classes = self.n_classes ).float()
        else:
            pose_parse = self.transform_mask(pose_parsing_pillow)
        """

        if( self.datamode in ["train", "valid"] ):
            results_dict = {
                "image_s_name" : pose_name,
                "image_t_gt_name" : pose_parsing_name,
                "image_s" : pose_parse,
                "image_t_gt" : pose_gt,
            }
        else:
            results_dict = {
                "image_name" : pose_parsing_name,
                "image_s" : pose_parse,
            }

        return results_dict


class ZalandoDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(ZalandoDataLoader, self).__init__()
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