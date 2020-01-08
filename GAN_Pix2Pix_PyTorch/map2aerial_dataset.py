# coding=utf-8
import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

class Map2AerialDataset(data.Dataset):
    """
    航空写真と地図画像のデータセットクラス
    """
    def __init__(self, root_dir, datamode = "train", image_height = 256, image_width = 256, debug = False ):
        super(Map2AerialDataset, self).__init__()

        # データをロードした後に行う各種前処理の関数を構成を指定する。
        self.transform = transforms.Compose(
            [
                transforms.Resize( (image_height, 2*image_width), interpolation=Image.LANCZOS ),
                transforms.CenterCrop( size = (image_height, 2 * image_width) ),
                transforms.ToTensor(),   # Tensor に変換
            ]
        )

        #
        self.image_height = image_height
        self.image_width = image_width
        self.dataset_dir = os.path.join( root_dir, datamode )
        self.image_names = sorted( [f for f in os.listdir(self.dataset_dir) if f.endswith(IMG_EXTENSIONS)] )
        self.debug = debug
        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        raw_image = Image.open(os.path.join(self.dataset_dir, image_name)).convert('RGB')
        #print( "raw_image.size",  raw_image.size )
        raw_image_tsr = self.transform(raw_image)
        #print( "raw_image_tsr.shape",  raw_image_tsr.shape )

        # 学習用データには、左側に衛星画像、右側に地図画像が入っているので、chunk で切り分ける
        # torch.chunk() : 渡したTensorを指定した個数に切り分ける。
        aerial_image_tsr, map_image_tsr = torch.chunk( raw_image_tsr, chunks=2, dim=2 )
        
        results_dict = {
            "image_name" : image_name,
            "raw_image_tsr" : raw_image_tsr,
            "aerial_image_tsr" : aerial_image_tsr,
            "map_image_tsr" : map_image_tsr,
        }
        return results_dict


class Map2AerialDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True):
        super(Map2AerialDataLoader, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle
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
