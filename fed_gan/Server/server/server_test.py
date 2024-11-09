# coding:utf-8
"""
    author:akun
    date : 12.10
    modify save fake
"""
import matplotlib.pyplot as plt
import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torchvision.transforms as transforms
import os

if __name__ == '__main__':
    opt = TestOptions().parse()  # get val options
    print("start generate fake_image")
    # opt.dataroot = '../data/ship/mask'
    opt.num_threads = 0   # val code only supports num_threads = 1
    opt.batch_size = 1    # val code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen trainB are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped trainB are needed.
    opt.display_id = -1   # no visdom display; the val code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    print(len(dataset))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test trainB.
            break
        # mask = torch.permute(data,[1,2,0]).detach.cpu().numpy()
        # plt.imshow(mask)
        # plt.show()
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visual = model.get_current_visuals()
        img_path = model.get_image_paths()

        # 1,3,256,256 -> 3,256,256
        fake = visual['fake'][0]

        fake = (fake+1)/2


        img_save = img_path[0].split('/')

        save_dir = "/".join(img_save[:-1])
        img_name = str(img_save[-1]).split('.')[0]
        img_name = img_name + '.jpg'

        save_path = os.path.join(save_dir,img_name)
        print(save_path)

        unloader = transforms.ToPILImage()
        fake_img = unloader(fake)
        fake_img.save(save_path)
