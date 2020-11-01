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

from utils import set_random_seed, onehot_encode_tsr

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class ZalandoDataset(data.Dataset):
    def __init__(self, args, root_dir, datamode = "train", image_height = 128, image_width = 128, n_classes = 20, data_augument = False, debug = False ):
        super(ZalandoDataset, self).__init__()
        self.args = args
        self.datamode = datamode
        self.data_augument = data_augument
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes
        self.debug = debug

        self.pose_dir = os.path.join( root_dir, "pose" )
        self.pose_parsing_dir = os.path.join( root_dir, "pose_parsing" )
        self.pose_names = sorted( [f for f in os.listdir(self.pose_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )
        self.pose_parsing_names = sorted( [f for f in os.listdir(self.pose_parsing_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )

        # transform
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(5.0, 5.0), scale = (0.80,1.25), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                ]
            )

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(5.0, 5.0), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                ]
            )

            self.transform_mask_woToTensor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(5.0, 5.0), scale = (0.80,1.25), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
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
            print( "len(self.pose_names) :", len(self.pose_names))
            print( "self.pose_names[0:5] :", self.pose_names[0:5])

        return

    def __len__(self):
        return len(self.pose_names)

    def __getitem__(self, index):
        pose_name = self.pose_names[index]

        # pose
        if( self.datamode in ["train", "valid"] ):
            pose_gt = Image.open( os.path.join(self.pose_dir,pose_name) ).convert('RGB')
            self.seed_da = random.randint(0,10000)
            if( self.data_augument ):
                set_random_seed( self.seed_da )

            pose_gt = self.transform(pose_gt)

        # pose paring
        pose_parsing_pillow = Image.open( os.path.join(self.pose_parsing_dir, pose_name) ).convert('L')

        self.seed_da = random.randint(0,10000)
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        pose_parse_onehot = torch.from_numpy( np.asarray(self.transform_mask_woToTensor(pose_parsing_pillow)).astype("int64") ).unsqueeze(0)
        pose_parse_onehot = onehot_encode_tsr( pose_parse_onehot, n_classes = self.n_classes ).float()

        if( self.datamode == "train" ):
            results_dict = {
                "pose_name" : pose_name,
                "pose_gt" : pose_gt,
                "pose_parse_onehot" : pose_parse_onehot,
            }
        else:
            results_dict = {
                "pose_name" : pose_name,
                "pose_parse_onehot" : pose_parse_onehot,
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