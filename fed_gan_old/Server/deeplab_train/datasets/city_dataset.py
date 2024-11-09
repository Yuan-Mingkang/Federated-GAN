import os.path
import random
import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform


class CityDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/images' contains image pairs in the form of {A,B}.
    During val time, you need to prepare a directory '/path/to/data/val'.
    """

    def __init__(self, opt, idx=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # if idx is None:
        #     opt.phase = 'images'
        #     # h5_name = "BraTS18_test.h5"
        # else:
        #     opt.phase = f"train_{idx}"
        #     # h5_name = f"BraTS18_tumor_size_{idx}.h5"
        #     print(f"Load: {opt.phase}")

        self.is_test = True
        self.real_tumor = False
        self.extend_len = 0
        self.multi_label = True

        BaseDataset.__init__(self, opt)
        # images
        file_dir = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        # val
        # file_dir = opt.dataroot
        file_names = os.listdir(file_dir)
        self.images = []
        self.masks = []

        for file_name in file_names:
            if file_name.endswith('_labels.png'):
                self.masks.append(os.path.join(file_dir, file_name))
                file_name = file_name.split('_')[0]
                file_name = str(file_name) + '_image.png'
                self.images.append(os.path.join(file_dir, file_name))
        assert (len(self.images) == len(self.masks))

        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
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
        A_path = self.masks[index]
        B_path = self.images[index]

        A = Image.open(A_path).convert('RGB')
        B = Image.open(B_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=Image.NEAREST)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), method=Image.NEAREST)

        A = A_transform(A)
        B = B_transform(B)

        # print('------------A')
        # print(list(A.size()))
        # print('------------B')
        # print(list(B.size()))


        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of trainB in the dataset."""
        return len(self.images)
# /media/ilab/aa4924cc-5cd3-4a4f-995c-4721b4a821ac/sbk/AsynDGAN-master/data