import torch
import torch.utils.data as data
import numpy as np
import os
from glob import glob
import os.path as osp
import random
from .frame_utils import *
from .augmentor import DisparityAugmentor

class BaseDataset(data.Dataset):
    def __init__(self, datapath, aug_params=None, test=False, overfit=False, seed=0, mono=None, multiplier=1, scale_factor=1.0, top_crop = 0):
        self.augmentor = DisparityAugmentor(**aug_params) if aug_params else None
        self.is_test = test
        self.seed = seed
        self.init_seed = False
        self.mono = mono
        self.multiplier = int(multiplier)
        self.scale_factor = scale_factor
        self.top_crop = top_crop
        self.calib_list = []

        self.image_list = []
        self.extra_info = []
        
        self.load_data(datapath)
        
        if overfit:
            self.image_list = self.image_list[:1]
            self.extra_info = self.extra_info[:1]
        
        if multiplier > 1:
            self.image_list *= multiplier
            self.extra_info *= multiplier
    
    def load_data(self, datapath):
        """To be implemented by subclasses."""
        raise NotImplementedError

    def __getitem__(self, index):
        if not self.init_seed and not self.is_test:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed((worker_info.id + self.seed) % 2**30)
                np.random.seed((worker_info.id + self.seed) % 2**30)
                random.seed((worker_info.id + self.seed) % 2**30)
                self.init_seed = True

        return self.load_sample(index)
    
    def load_sample(self, index):
        """To be implemented by subclasses."""
        raise NotImplementedError
    
    def __len__(self):
        return len(self.image_list)