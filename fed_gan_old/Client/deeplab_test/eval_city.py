from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
#
from torch.utils import data
from datasets import CitySegmentation
from torchvision import transforms as T
from metrics import StreamSegMetrics
import torchvision.transforms as transforms

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

# /media/ilab/aa4924cc-5cd3-4a4f-995c-4721b4a821ac/sbk/data/dota_isaid_800/val/images
def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    # parser.add_argument("--input", type=str, required=True,
    #                     help="path to a single images_beifen or images_beifen directory")
    parser.add_argument("--file_dir", type=str,
                        help="path to a single images_beifen or images_beifen directory")
    parser.add_argument("--num_classes", type=int,
                        help="path to a single images_beifen or images_beifen directory")
    parser.add_argument("--dataset", type=str, default='city',
                        choices=['city', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=None,
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='1',
                        help="GPU ID")
    return parser


def main():
    opts = get_argparser().parse_args()

    opts.file_dir = '/home/poac/4TB/yuanmingkang/dataset/city/gan/test/ALL512'
    opts.dataset = "city"
    opts.model = "deeplabv3plus_resnet101"

    opts.ckpt = f"/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab _cityclassify_train/city_fed_gan/checkpoints/deeplab_city/958_deeplabv3plus_resnet101_city_os16.pth"
    # opts.ckpt = f"/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab _cityclassify_train/city_gan_lr/checkpoints/deeplab_city/958_deeplabv3plus_resnet101_city_os16.pth"
    # opts.ckpt = f"/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab _cityclassify_train/city_real/checkpoints/deeplab_city/958_deeplabv3plus_resnet101_city_os16.pth"

    # opts.save_val_results_to = "/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab_city_test/city_fed_gan/result"
    # opts.save_val_results_to = "/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab_city_test/city_gan_lr/result"
    # opts.save_val_results_to = "/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab_city_test/epoch987/pred_real"

    # class 0: 0.6266901015193603
    #
    # class 1: 0.4639769246509087
    #
    # class 2: 0.536554982481449
    #
    # miou: 0.542407336217239
    # pa: 0.7155358794796645
    # dice: 0.700686225016932


    # double d
    # opts.ckpt = "./checkpoints_2fake/latest_deeplabv3plus_resnet101_city_os16.pth"
    # opts.save_val_results_to = "./results_2fake"

    # single d
    # opts.ckpt = "./checkpoints_city/latest_deeplabv3plus_resnet101_city_os16.pth"
    # opts.save_val_results_to = "./results_1fake_new"


    # real images
    # opts.ckpt = "./checkpoints_city/latest_deeplabv3plus_resnet101_city_os16.pth"
    # opts.save_val_results_to = "./results_real"


    opts.crop_size = 512 # gai 512
    opts.val_batch_size = 1
    opts.num_classes = 3
    opts.crop_val

    if opts.dataset.lower() == 'city':
        decode_fn = CitySegmentation.decode_target
        encode_fn = CitySegmentation.encode_target

    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Setup dataloader
    images_files = []
    labels_files = []

    labels_name = os.listdir(opts.file_dir)
    for label_name in labels_name:
        if label_name.endswith('_labels.png'):
            labels_files.append(os.path.join(opts.file_dir,label_name))
            image_name = label_name.split('_')[0]
            image_name = str(image_name)+'_image.png'
            images_files.append(os.path.join(opts.file_dir,image_name))
    print(f"dataset have {len(images_files)} images")

    # Set up model
    model_map = {
        'deeplabv3_resnet50': network.deeplabv3_resnet50,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3_resnet101': network.deeplabv3_resnet101,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }


    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        # Load the pretrained model
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images_beifen

    if opts.crop_val:
        transform = T.Compose([
            # T.Resize(opts.crop_size),
            # T.CenterCrop(opts.crop_size),
            T.Resize(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    miou = 0.0
    pa = 0.0
    dice = 0.0
    cls_iou = {}
    cls_iou[0] = 0.0
    cls_iou[1] = 0.0
    cls_iou[2] = 0.0
    i = 0
    with torch.no_grad():
        model = model.eval()
        metrics.reset()
        preds = []
        labels = []

        for label_path,img_path in zip(labels_files,images_files):
            i = i + 1
            img_name = os.path.basename(img_path).split('.')[0]

            img = Image.open(img_path).convert('RGB')
            label = Image.open(label_path)
            # print(img.size)
            # print(label.size)
            label = encode_fn(label)

            # resize
            resize = transforms.Resize([512, 512], Image.BICUBIC)
            label = Image.fromarray(label)
            # img = Image.fromarray(img)
            label = resize(label)
            img = resize(img)
            label = np.array(label)
            img = np.array(img)
            # label = transform(label)

            img = transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(device)
            # Visualize segmentation outputs:
            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW




            # print(f"pred {pred.size}")
            # print(f"label {label.size}")
            # preds.append(pred)
            # labels.append(label)
            # do what you want to do here



            metrics.update(label, pred)
            score = metrics.get_results()
            miou = miou + float(score['Mean IoU'])
            pa = pa + float(score['Overall Acc'])
            dice = dice + float(score['Mean Dice'])
            class_iou = score['Class IoU']
            for k , v in class_iou.items():
                cls_iou[k] = cls_iou[k]+ float(v)

            # for index,cls_one in score['Mean IoU']:
            #     cls_iou


            # # print(metrics.to_str(score))
            if i % 100 == 0:
                print(f"process image : {str(i)}")
            #     break
            # plt pred
            # pred = decode_fn(pred)
            # predimg = Image.fromarray(np.uint8(pred))
            # predimg.show()
            # plt label
            # label = decode_fn(label)
            # lableimg = Image.fromarray(np.uint8(label))
            # lableimg.show()
            pred = decode_fn(pred)
            pred_save = Image.fromarray(np.uint8(pred))

            # 展示图片
            # print("-----show images_beifen——————")
            # plt.title(str(i))
            # plt.imshow(pred_save)
            # plt.show()
            # i = i + 1
            # print(str(i))
            if opts.save_val_results_to:

                pred_save.save(os.path.join(opts.save_val_results_to, img_name + '_pred.png'))

    for i in range(0,opts.num_classes):
        print(f"class {i}:  {cls_iou[i]/len(images_files)}")
    print("finish")
    print('miou:',miou/len(images_files))
    print('pa:' , pa/len(images_files))
    print('dice:' , dice/len(images_files))



if __name__ == '__main__':
    main()
# 34045