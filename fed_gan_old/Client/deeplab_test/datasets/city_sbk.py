import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from collections import namedtuple

class CitySegmentation(data.Dataset):
    """
        Args:
            root: dataset root where have two subfolder （images/val）
            image_set: images or val
            transform:
    """
    Label = namedtuple('Label', [
        'id',
        'color',  # The color of this label
        'm_color',  # The color of this label
    ])
    labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color          multiplied color
        Label( 0, (255, 255,255), 1020),
        Label(1,  (255, 0, 0), 255),
        Label(2, (0, 0, 255), 510),
    ]

    # m2id = {label.m_color: label.id for label in labels}

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
        self.masks = []
        for file_name in file_names:
            if file_name.endswith('_labels.png'):
                self.masks.append(os.path.join(file_dir, file_name))
                file_name = file_name.split('_')[0]
                file_name = str(file_name) + '_image.png'
                self.images.append(os.path.join(file_dir, file_name))
        assert (len(self.images) == len(self.masks))

    @classmethod
    def encode_target(cls, target):

        # rgb to single
        target = np.array(target)
        target = target.astype('uint16')
        target = target[:, :, 0] + target[:, :, 1] + 2 * target[:, :, 2]
        target[target == 255] = 0  # red
        target[target == 510] = 1  # blue
        target[target == 1020] = 2  # white
        target = target.astype('uint8')
        # print(target.shape)

        # resize
        target = Image.fromarray(target)
        resize = transforms.Resize([512, 512], Image.BICUBIC)
        target = resize(target)
        target = np.array(target)
        return target

    @classmethod
    def decode_target(cls, target):
        # target = np.array(target)
        # target[target == 255] = 15
        # target = target.astype('uint8') + 1
        # return cls.trainid2color[target]

        # single to rgb
        target = np.array(target)
        target = target[:, :, np.newaxis]
        target = np.repeat(target, 3, axis=2)
        target[target[:, :, 0] == 0] = [255, 0, 0]  # red
        target[target[:, :, 0] == 1] = [0, 0, 255]  # blue
        target[target[:, :, 0] == 2] = [255, 255, 255]  # white
        return target

    def __getitem__(self, index):
        """
        :param index:
        :return: img ,target
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        # image = np.array(target)
        target = self.encode_target(target)
        target = Image.fromarray(target)
        # print(target.size())

        img = self.transform(img)
        target  = self.transform(target)
        # if self.transform is not None:
        #     img, target = self.transform(img, target)
        print(img.size())
        print(target.size())
        return img, target

    def __len__(self):
        return len(self.images)
