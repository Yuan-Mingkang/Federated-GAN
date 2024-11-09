import torch.utils.data as data
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# add 1 1024
class CitySegmentation(data.Dataset):
    """
        Args:
            root: dataset root where have two subfolder （images/val）
            image_set: images or val
            transform:
    """
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

        # rgb to single
        target = np.array(target)
        target = target.astype('uint16')
        target = target[:, :, 0] + target[:, :, 1] + 2 * target[:, :, 2]

        target[target == 255] = 0  # red
        target[target == 510] = 1  # blue
        target[target == 1020] = 2  # white

        target = target.astype('uint8')

        # resize
        # target = Image.fromarray(target)
        # resize = transforms.Resize([512, 512], Image.BICUBIC)
        # target = resize(target)
        # target = np.array(target)

        return target

    @classmethod
    def decode_target(cls, target):

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

        # resize mask as the size of img
        # if self.image_set == 'train':
        #     target = Image.fromarray(target)
        #     # resize = transforms.Resize(img.size, Image.BICUBIC)
        #     # target = resize(target)
        #     target = np.array(target)
        # else:
        #     target = Image.fromarray(target)
        #     resize = transforms.Resize(img.size, Image.BICUBIC)
        #     target = resize(target)
        #     target = np.array(target)

        target = Image.fromarray(target)

        if self.transform is not None:
            # img = self.transform(img)
            # target = self.transform(target)

            img, target = self.transform(img, target)
        return img, target, self.masksname

    def __len__(self):
        return len(self.images)
