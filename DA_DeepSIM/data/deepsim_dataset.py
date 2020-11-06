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
from utils import set_random_seed, onehot_encode_tsr, numerical_sort

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class DeepSIMDataset(data.Dataset):
    def __init__(self, args, dataset_dir, datamode = "train", data_type = "car", image_height = 128, image_width = 128, n_classes = 20, data_augument_type = "none", onehot = False, debug = False ):
        super(DeepSIMDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.datamode = datamode
        self.data_type = data_type
        self.data_augument_type = data_augument_type
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes
        self.onehot = onehot
        self.debug = debug

        self.img_A_train_dir = os.path.join( self.dataset_dir, self.data_type, "train_A" )
        self.img_B_train_dir = os.path.join( self.dataset_dir, self.data_type, "train_B" )
        self.img_A_test_dir = os.path.join( self.dataset_dir, self.data_type, "test_A" )

        self.img_A_train_names = sorted( [f for f in os.listdir(self.img_A_train_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        self.img_B_train_names = sorted( [f for f in os.listdir(self.img_B_train_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        self.img_A_test_names = sorted( [f for f in os.listdir(self.img_A_test_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )

        # transform
        if( data_augument_type == "none" ):
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
        elif( data_augument_type == "affine" ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                ]
            )

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                ]
            )

            self.transform_mask_woToTensor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                ]
            )
        elif( data_augument_type == "affine_tps" ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    TPSTransform( tps_points_per_dim = self.args.tps_points_per_dim ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                ]
            )

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    TPSTransform( tps_points_per_dim = self.args.tps_points_per_dim ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                ]
            )

            self.transform_mask_woToTensor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.25,0.25), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    TPSTransform( tps_points_per_dim = self.args.tps_points_per_dim ),
                ]
            )
        elif( data_augument_type == "full" ):
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
            NotImplementedError()

        if( self.debug ):
            print( "self.img_A_train_names : ", self.img_A_train_names )
            print( "self.img_B_train_names : ", self.img_B_train_names )
            print( "self.img_A_test_names : ", self.img_A_test_names )

        return

    def __len__(self):
        if( self.datamode in ["train", "valid"] ):
            return len(self.img_A_train_names)
        else:
            return len(self.img_A_test_names)

    def __getitem__(self, index):
        if( self.datamode in ["train", "valid"] ):
            img_A_name = self.img_A_train_names[index]
            img_B_name = self.img_B_train_names[index]
        else:
            img_A_name = self.img_A_test_names[index]

        self.seed_da = random.randint(0,10000)

        # img A
        if( self.datamode in ["train", "valid"] ):
            imgA_pillow = Image.open( os.path.join(self.img_A_train_dir, img_A_name) ).convert('RGB')
        else:
            imgA_pillow = Image.open( os.path.join(self.img_A_test_dir, img_A_name) ).convert('RGB')

        if( self.data_augument_type != "none" ):
            set_random_seed( self.seed_da )

        if( self.onehot ):
            imgA = torch.from_numpy( np.asarray(self.transform_mask_woToTensor(imgA_pillow)).astype("int64") ).unsqueeze(0)
            print( "imgA.shape : ", imgA.shape )
            imgA = onehot_encode_tsr( imgA, n_classes = self.n_classes ).float()
        else:
            imgA = self.transform(imgA_pillow)

        # img B
        if( self.datamode in ["train", "valid"] ):
            imgB_gt = Image.open( os.path.join(self.img_B_train_dir, img_B_name) ).convert('RGB')
            if( self.data_augument_type != "none" ):
                set_random_seed( self.seed_da )

            imgB_gt = self.transform(imgB_gt)

        if( self.datamode in ["train", "valid"] ):
            results_dict = {
                "image_s_name" : img_B_name,
                "image_t_gt_name" : img_A_name,
                "image_s" : imgA,
                "image_t_gt" : imgB_gt,
            }
        else:
            results_dict = {
                "image_s_name" : img_A_name,
                "image_s" : imgA,
            }

        return results_dict


class DeepSIMDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(DeepSIMDataLoader, self).__init__()
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