import os
import numpy as np
import random
import pandas as pd
import numbers
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

from utils import set_random_seed

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class CIHPDataset(data.Dataset):
    def __init__(self, args, root_dir, datamode = "train", flip = False, data_augument = False, debug = False ):
        super(CIHPDataset, self).__init__()
        self.args = args
        self.datamode = datamode
        self.data_augument = data_augument
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.flip = flip
        self.debug = debug
        self.image_dir = os.path.join( root_dir, "Images" )
        self.categories_dir = os.path.join( root_dir, "Categories" )
        self.categories_rev_dir = os.path.join( root_dir, "Category_rev_ids" )

        self.image_names = []
        self.categories_names = []
        self.categories_rev_names = []
        with open( os.path.join(root_dir, "lists", datamode + '_id.txt'), "r" ) as f:
            lines = f.read().splitlines()
            for _, line in enumerate(lines):
                image_name = os.path.join(self.image_dir, line+'.jpg' )
                categories_name = os.path.join(self.categories_dir, line +'.png')
                categories_rev_name = os.path.join(self.categories_rev_dir,line + '.png')
                assert os.path.isfile(image_name)
                assert os.path.isfile(categories_name)
                assert os.path.isfile(categories_rev_name)
                self.image_names.append(image_name)
                self.categories_names.append(categories_name)
                self.categories_rev_names.append(categories_rev_name)

        assert (len(self.image_names) == len(self.categories_names))
        assert len(self.categories_rev_names) == len(self.categories_names)
        
        # transform
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
#                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
#                    transforms.RandomVerticalFlip(),
#                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5,0.5,0.5], [0.5,0.5,0.5] ),
                ]
            )

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
#                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
#                    transforms.RandomVerticalFlip(),
#                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                ]
            )

            self.transform_mask_woToTensor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
#                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
#                    transforms.RandomVerticalFlip(),
#                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.NEAREST ),
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
            print( "self.image_dir :", self.image_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])

        return

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        categories_name = self.categories_names[index]
        categories_rev_name = self.categories_rev_names[index]

        # image
        image = Image.open(image_name).convert('RGB')
        if( self.flip ):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        self.seed_da = random.randint(0,10000)
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        image = self.transform(image)

        # Categories_ids
        if( self.flip ):
            target = Image.open(categories_rev_name).convert("L")
        else:
            target = Image.open(categories_name).convert("L")

        self.seed_da = random.randint(0,10000)
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        #print( "[np] target : ", np.asarray(target).astype("int64") )        
        if( self.args.n_output_channels == 1 ):
            target = self.transform_mask(target)
        else:
            #target = self.transform_mask(target)
            #target = torch.from_numpy( np.asarray(self.transform_mask_woToTensor(target)).astype("float32").transpose(2,0,1) )
            target = torch.from_numpy( np.asarray(self.transform_mask_woToTensor(target)).astype("float32") ).unsqueeze(0)

        if( self.datamode == "train" ):
            results_dict = {
                "image" : image,
                "target" : target,
            }
        else:
            results_dict = {
                "image" : image,
            }

        return results_dict


class CIHPDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(CIHPDataLoader, self).__init__()
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