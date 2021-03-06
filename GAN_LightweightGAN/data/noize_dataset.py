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

class NoizeDataset(data.Dataset):
    def __init__(self, args, root_dir, datamode = "train", image_size = 1024, z_dims = 256, debug = False ):
        super(NoizeDataset, self).__init__()
        self.args = args
        self.datamode = datamode
        self.image_size = image_size
        self.z_dims = z_dims
        self.debug = debug

        self.image_t_dir = os.path.join( root_dir )
        self.image_t_names = sorted( [f for f in os.listdir(self.image_t_dir) if f.endswith(IMG_EXTENSIONS)], key=numerical_sort )
        self.latent_z_fixed = self.get_latent_z(z_dims)

        self.transform = transforms.Compose(
            [
                transforms.Resize( (args.image_size, args.image_size), interpolation=Image.LANCZOS ),
                transforms.CenterCrop( size = (args.image_size, args.image_size) ),
                transforms.ToTensor(),
                transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
            ]
        )
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize( (args.image_size, args.image_size), interpolation=Image.NEAREST ),
                transforms.CenterCrop( size = (args.image_size, args.image_size) ),
                transforms.ToTensor(),
#                    transforms.Normalize( [0.5], [0.5] ),
                transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
            ]
        )
        self.transform_mask_woToTensor = transforms.Compose(
            [
                transforms.Resize( (args.image_size, args.image_size), interpolation=Image.NEAREST ),
                transforms.CenterCrop( size = (args.image_size, args.image_size) ),
            ]
        )

        if( self.debug ):
            print( "self.image_t_dir :", self.image_t_dir)
            print( "self.image_t_names[0:5] :", self.image_t_names[0:5])

        return

    def __len__(self):
        return len(self.image_t_names)

    def get_latent_z(self, z_dims, noize_type = "uniform" ):
        if( noize_type == "uniform" ):
            #latent_z = torch.randn([z_dims])
            latent_z = torch.Tensor(z_dims).normal_(0, 1)
        else:
            NotImplementedError()

        return latent_z

    def __getitem__(self, index):
        image_t_name = self.image_t_names[index]

        #---------------------
        # input noize z
        #---------------------
        if( self.datamode == "train" ):
            latent_z = self.get_latent_z(self.z_dims)
        else:
            latent_z = self.latent_z_fixed

        #---------------------
        # image_t
        #---------------------
        if( self.datamode == "train" ):
            image_t = Image.open( os.path.join(self.image_t_dir, image_t_name) )
            image_t = self.transform(image_t)

        #---------------------
        # returns
        #---------------------
        if( self.datamode == "train" ):
            results_dict = {
                "latent_z" : latent_z,
                "image_t_name" : image_t_name,
                "image_t" : image_t,
            }
        else:
            results_dict = {
                "latent_z" : latent_z,
            }

        return results_dict

