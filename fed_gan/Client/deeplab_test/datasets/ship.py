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
    # def __init__(self,
    #              root,
    #              image_set='images',
    #              transform=None):
    #
    #     self.transform = transform
    #     file_dir = os.path.join(root)
    #     file_names = sorted(os.listdir(file_dir))
    #     file_nums = len(file_names)*0.5
    #     print(f"len_dataset{file_nums}")
    #     self.images = []
    #     self.masks = []
    #     masks = []
    #     images = []
    #
    #     for file_name in file_names:
    #         if file_name.endswith('.png'):
    #             masks.append(os.path.join(file_dir, file_name))
    #             file_name = file_name.split('.')[0]
    #             file_name = str(file_name) + '.jpg'
    #             images.append(os.path.join(file_dir, file_name))
    #     assert (len(self.images) == len(self.masks))
    #     if image_set == 'images':
    #         self.masks = masks[:int(file_nums*0.8)]
    #         self.images = images[:int(file_nums*0.8)]
    #     elif image_set == 'val':
    #         self.masks = masks[int(file_nums * 0.8):]
    #         self.images = images[int(file_nums * 0.8):]
    #     print(f"{image_set} have img_num {self.masks}")
    def __init__(self,
                 mask_path,
                 train_path,
                 image_set='train',
                 transform=None):

        self.image_set = image_set
        self.transform = transform
        mask_dir = os.path.join(mask_path)
        mask_names = sorted(os.listdir(mask_dir))
        mask_nums = len(mask_names)

        train_dir = os.path.join(train_path)
        train_names = sorted(os.listdir(train_dir))
        train_nums = len(train_names)
        # print(f"len_dataset{train_nums}")

        # file_dir = os.path.join(root)
        # file_names = sorted(os.listdir(file_dir))
        # file_nums = len(file_names)*0.5
        # print(f"len_dataset{file_nums}")
        self.images = []
        self.masks = []
        masks = []
        images = []
        self.masksname = []

        for mask_name in mask_names:
            masks.append(os.path.join(mask_dir, mask_name))
            self.masksname.append(mask_name)
        for train_name in train_names:
            images.append(os.path.join(train_dir, train_name))
        # for file_name in file_names:
        #     if file_name.endswith('.png'):
        #         masks.append(os.path.join(file_dir, file_name))
        #         file_name = file_name.split('.')[0]
        #         file_name = str(file_name) + '.jpg'
        #         images.append(os.path.join(file_dir, file_name))
        assert (len(self.images) == len(self.masks))
        if image_set == 'train':
            self.masks = masks
            self.images = images
            # self.masks = masks[:int(mask_nums*0.8)]
            # self.images = images[:int(train_nums*0.8)]
            self.masks = masks[:int(mask_nums * 1.0)]
            self.images = images[:int(train_nums * 1.0)]
        elif image_set == 'val':
            # self.masks = masks[int(mask_nums * 0.8):]
            # self.images = images[int(train_nums * 0.8):]
            self.masks = masks[int(mask_nums * 1):]
            self.images = images[int(train_nums * 1):]
        print(f"{image_set} have img_num {len(self.masks)}")
    @classmethod
    def encode_target(cls, target):
        target = np.array(target)
        target = target.astype('uint16')
        target = target[:, :, 0] + target[:, :, 1] + 2 * target[:, :, 2]

        target[target == 0] = 0
        target[target == 1020] = 1
        target = target.astype('uint8')
        # target[target == 0] = 0
        # target[target == 255] = 1
        # target[target == 0] =
        return target

    @classmethod
    def decode_target(cls, target):
        target = np.array(target)
        target = target[:, :, np.newaxis]
        target = np.repeat(target, 3, axis=2)
        target[target[:, :, 0] == 0] = [0, 0, 0]  # red
        target[target[:, :, 0] == 1] = [255, 255, 255]  # white
        # target = np.array(target)
        #
        # # target[target == 255] = 0
        # target[target == 1] = 255
        # # target[target == 0] = 0
        # target = np.array([target for i in range(3)]).transpose(1, 2, 0)
        return target
    def __getitem__(self, index):
        """
        :param index:
        :return: img ,target
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index]).convert('RGB')
        # print(target.size)
        # image = np.array(target)
        target = self.encode_target(target)
        target = Image.fromarray(target)


        if self.transform is not None:
            img, target = self.transform(img, target)
        # print(img.size())
        return img, target, self.masksname

    def __len__(self):
        return len(self.images)
