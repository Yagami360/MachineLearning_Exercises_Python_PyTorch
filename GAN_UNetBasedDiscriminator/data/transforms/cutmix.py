import numpy as np
from PIL import Image
from scipy import ndimage

import torch
import torchvision.transforms as transforms
from utils import set_random_seed

class CutMix(object):
    def __init__(self, prob = 0.5, lam = 1 ):
        self.prob = prob
        self.lam = lam
        return

    def set_seed(self, seed=12):
        set_random_seed(seed)
        return

    def random_boundingbox(self, height, width):
        r = np.sqrt(1. - np.random.beta(self.lam,self.lam))
        w = np.int(width * r)
        h = np.int(height * r)
        x = np.random.randint(width)
        y = np.random.randint(height)
        x1 = np.clip(x - w // 2, 0, width)
        y1 = np.clip(y - h // 2, 0, height)
        x2 = np.clip(x + w // 2, 0, width)
        y2 = np.clip(y + h // 2, 0, height)
        return x1, y1, x2, y2

    def __call__(self, image_s, image_t):
        h, w = image_s.shape[2], image_s.shape[3]
        x1, y1, x2, y2 = self.random_boundingbox(h,w)

        cutmix_mask = torch.ones((h,w))
        cutmix_mask[x1:x2,y1:y2] = 0
        if torch.rand(1) > self.prob:
            cutmix_mask = 1 - cutmix_mask

        # マスク画像を元に画像を mix
        image_s = cutmix_mask * image_s + ( 1 - cutmix_mask ) * image_t
        return image_s, cutmix_mask

