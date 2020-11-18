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

class TempleteDataset(data.Dataset):
    def __init__(self, args, root_dir, datamode = "train", image_height = 128, image_width = 128, data_augument = False, debug = False ):
        super(TempleteDataset, self).__init__()
        self.args = args
        self.datamode = datamode
        self.data_augument = data_augument
        self.image_height = image_height
        self.image_width = image_width
        self.debug = debug

        self.image_s_dir = os.path.join( root_dir, "image_s" )
        self.image_t_dir = os.path.join( root_dir, "image_t" )
        #self.image_s_names = sorted( [f for f in os.listdir(self.image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )
        #self.image_t_names = sorted( [f for f in os.listdir(self.image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )
        self.image_s_names = sorted( [f for f in os.listdir(self.image_s_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        self.image_t_names = sorted( [f for f in os.listdir(self.image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )

        # transform
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
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
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.NEAREST ),
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
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.NEAREST ),
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
#                    transforms.Normalize( [0.5], [0.5] ),
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
            print( "self.image_s_dir :", self.image_s_dir)
            print( "self.image_t_dir :", self.image_t_dir)
            print( "len(self.image_s_names) :", len(self.image_s_names))
            print( "self.image_s_names[0:5] :", self.image_s_names[0:5])

        return

    def __len__(self):
        return len(self.image_s_names)

    def __getitem__(self, index):
        image_s_name = self.image_s_names[index]
        image_t_name = self.image_t_names[index]
        self.seed_da = random.randint(0,10000)

        #---------------------
        # image_s
        #---------------------
        image_s = Image.open( os.path.join(self.image_s_dir,image_s_name) ).convert('RGB')
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        image_s = self.transform(image_s)

        #---------------------
        # image_t
        #---------------------
        if( self.datamode == "train" ):
            #image_t = Image.open( os.path.join(self.image_t_dir, image_t_name) )
            image_t = Image.open( os.path.join(self.image_t_dir, image_t_name) ).convert('RGB')
            #self.seed_da = random.randint(0,10000)
            if( self.data_augument ):
                set_random_seed( self.seed_da )

            image_t = self.transform_mask(image_t)
            #image_t = torch.from_numpy( np.asarray(self.transform_mask_woToTensor(image_t)).astype("float32") ).unsqueeze(0)

        #---------------------
        # returns
        #---------------------
        if( self.datamode == "train" ):
            results_dict = {
                "image_s_name" : image_s_name,
                "image_t_name" : image_t_name,
                "image_s" : image_s,
                "image_t" : image_t,
            }
        else:
            results_dict = {
                "image_s_name" : image_s_name,
                "image_s" : image_s,
            }

        return results_dict


class TempleteDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(TempleteDataLoader, self).__init__()
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