import torch.utils.data as data
import os
from PIL import Image
import numpy as np
class ShipSegmentation(data.Dataset):
    """
        Args:
            root: dataset root where have many images
            image_set: images or val
            transform:
        modify akun 2021..11.24
        add split images val
    """
    def __init__(self,
                 root,
                 image_set='images',
                 transform=None):

        self.transform = transform
        file_dir = os.path.join(root)
        file_names = sorted(os.listdir(file_dir))
        file_nums = len(file_names)*0.5
        # print(f"len_dataset{file_nums}")
        self.images = []
        self.masks = []
        masks = []
        images = []

        for file_name in file_names:
            if file_name.endswith('.png'):
                masks.append(os.path.join(file_dir, file_name))
                file_name = file_name.split('.')[0]
                file_name = str(file_name) + '.jpg'
                images.append(os.path.join(file_dir, file_name))
        assert (len(self.images) == len(self.masks))
        if image_set == 'images':
            self.masks = masks[:int(file_nums*0.8)]
            self.images = images[:int(file_nums*0.8)]
        elif image_set == 'val':
            self.masks = masks[int(file_nums * 0.8):]
            self.images = images[int(file_nums * 0.8):]
        # print(f"{image_set} have img_num {self.masks}")
    @classmethod
    def encode_target(cls, target):
        target = np.array(target)
        target[target == 255] = 1
        # target[target == 0] =
        return target

    @classmethod
    def decode_target(cls, target):
        target = np.array(target)
        # target[target == 255] = 0
        target[target == 1] = 255
        target = np.array([target for i in range(3)]).transpose(1, 2, 0)
        return target
    def __getitem__(self, index):
        """
        :param index:
        :return: img ,target
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        # print(target.size)
        # image = np.array(target)
        target = self.encode_target(target)
        target = Image.fromarray(target)

        if self.transform is not None:
            img, target = self.transform(img, target)
        # print(img.size())
        # print(target.size())
        return img, target

    def __len__(self):
        return len(self.images)
