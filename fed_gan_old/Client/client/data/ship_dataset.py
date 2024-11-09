import os.path
import random
import sys

import cv2
import matplotlib.pyplot as plt

import numpy as np
import torch

from PIL import Image
#from data.image_folder import make_dataset
from .base_dataset import BaseDataset, get_params, get_transform


class ShipDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, idx=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if idx is None:
            opt.phase = 'train'
            # h5_name = "BraTS18_test.h5"
        else:
            opt.phase = f"train_{idx}"
            # h5_name = f"BraTS18_tumor_size_{idx}.h5"
            print(f"Load: {opt.phase}")

        self.is_test = True
        self.real_tumor = False
        self.extend_len = 0
        self.multi_label = True

        BaseDataset.__init__(self, opt)
        # train
        file_dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # test
        file_names = sorted(os.listdir(file_dir))

        self.images = []
        self.masks = []

        for file_name in file_names:
            if file_name.endswith('.png'):
                mask_name = os.path.join(file_dir, file_name)
                self.masks.append(mask_name)    #'xxx.png' is mask
                file_name = file_name.split('.')[0]

                file_name = str(file_name) + '.jpg' #'xxx.jpg'is real image
                img_name = os.path.join(file_dir, file_name)
                self.images.append(img_name)

        assert (len(self.images) == len(self.masks))

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        index = idx[0]
        transform_params_server = idx[1]
        A_path = self.masks[index]
        B_path = self.images[index]
        B_name = B_path.split('/')[-1]
        # if index == 0:
        #     mask = Image.open(A_path)
        #     mask.save('/home/ilab/myself/project/Client/data/0.png')

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        A_transform = get_transform(self.opt, transform_params_server, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params_server, grayscale=(self.output_nc == 1), method=Image.NEAREST)
        A = A_transform(A)
        B = B_transform(B)
        # img = torch.permute(B,[1,2,0]).detach().cpu().numpy()
        # plt.imshow(img)
        # plt.show()

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'B_names':B_name}

    def __len__(self):
        """Return the total number of trainB in the dataset."""
        return len(self.images)