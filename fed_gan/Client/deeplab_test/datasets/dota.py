import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from collections import namedtuple

import natsort


class DOTASegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``images``, ``trainval`` or ``val``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL images_beifen
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    Label = namedtuple('Label', [

        'name',  # The identifier of this label, e.g. 'car', 'person', ... .
        # We use them to uniquely name a class

        'id',  # An integer ID that is associated with this label.
        # The IDs are used to represent the label in ground truth images_beifen
        # An ID of -1 means that this label does not have an ID and thus
        # is ignored when creating ground truth images_beifen (e.g. license plate).
        # Do not modify these IDs, since exactly these IDs are expected by the
        # evaluation server.

        'trainId',  # Feel free to modify these IDs as suitable for your method. Then create
        # ground truth images_beifen with images IDs, using the tools provided in the
        # 'preparation' folder. However, make sure to validate or submit results
        # to our evaluation server using the regular IDs above!
        # For trainIds, multiple labels might have the same ID. Then, these labels
        # are mapped to the same class in the ground truth images_beifen. For the inverse
        # mapping, we use the label that is defined first in the list below.
        # For example, mapping all void-type classes to the same ID in training,
        # might make sense for some approaches.
        # Max value is 255!

        'category',  # The name of the category that this label belongs to

        'categoryId',  # The ID of this category. Used to create ground truth images_beifen
        # on category level.

        'hasInstances',  # Whether this label distinguishes between single instances or not

        'ignoreInEval',  # Whether pixels having this class as ground truth label are ignored
        # during evaluations or not

        'color',  # The color of this label
        'm_color',  # The color of this label
    ])
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color          multiplied color
        Label('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0), 0),
        Label('Small_Vehicle', 1, 0, 'transport', 1, True, False, (0, 0, 127), 8323072),
        Label('Harbor', 2, 1, 'transport', 1, True, False, (0, 100, 155), 10183680),
    ]

    m2id = {label.m_color: label.id for label in labels}

    trainid2color = [label.color for label in labels]

    trainid2color = np.array(trainid2color)

    def __init__(self,
                 root,
                 image_set='images',
                 transform=None):

        self.transform = transform
        file_dir = os.path.join(root, image_set)


        file_names = os.listdir(file_dir)
        self.images = []
        self.masks= []

        if image_set == 'images':
            for file_name in file_names:
                if file_name.endswith('_AB_real_A.png'):
                    self.masks.append(os.path.join(file_dir, file_name))

                    file_name = file_name.split('_')[:-3]

                    if len(file_name) > 1:
                        file_name = '_'.join(file_name)
                    else:
                        file_name = ''.join(file_name)
                    file_name = str(file_name) + '_AB_fake_B.png'
                    self.images.append(os.path.join(file_dir, file_name))
        elif image_set == 'val':
            for file_name in file_names:
                if file_name.endswith('_instance_color_RGB.png'):
                    self.masks.append(os.path.join(file_dir, file_name))

                    file_name = file_name.split('_')[:-3]

                    if len(file_name) > 1:
                        file_name = '_'.join(file_name)
                    else:
                        file_name = ''.join(file_name)
                    file_name = str(file_name) + '.png'
                    self.images.append(os.path.join(file_dir, file_name))

        assert (len(self.images) == len(self.masks))


    @classmethod
    def encode_target(cls, target):
        imgarr =np.array(target)
        imgarrnp = imgarr[:, :, 0] + 256 * imgarr[:, :, 1] + 256 * 256 * imgarr[:, :, 2]
        img = np.zeros_like(imgarrnp)
        for k in cls.m2id:
            img[imgarrnp == k] = cls.m2id[k]
        return img

    @classmethod
    def decode_target(cls, target):
        target = np.array(target)
        # target[target == 255] = 15
        # target = target.astype('uint8') + 1
        return cls.trainid2color[target]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (images_beifen, target) where target is the images_beifen segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        target = self.encode_target(target)
        target = Image.fromarray(target)

        if self.transform is not None:
            img, target = self.transform(img, target)

        print(img.size())
        print(target.size())

        return img, target

    def __len__(self):
        return len(self.images)





