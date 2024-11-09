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
from datasets import ShipSegmentation
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
    parser.add_argument("--dataset", type=str, default='ship',
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

    opts.file_dir = '/home/poac/4TB/yuanmingkang/dataset/WHU_Building/Satellite dataset вё (global cities)/test/ALL512'
    opts.dataset = "ship"
    opts.model = "deeplabv3plus_resnet101"

    # opts.save_val_results_to = "/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab_ship_test/fed_gan/result"

    opts.crop_size = 512 # gai 512
    opts.val_batch_size = 1
    opts.num_classes = 2
    opts.crop_val

    if opts.dataset.lower() == 'ship':
        decode_fn = ShipSegmentation.decode_target
        encode_fn = ShipSegmentation.encode_target

    # set gpus
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Setup dataloader
    images_files = []
    labels_files = []
    labels_names = []

    labels_name = os.listdir(opts.file_dir)
    for label_name in labels_name:
        if label_name.endswith('label.tif'):
            labels_files.append(os.path.join(opts.file_dir, label_name))
            labels_names.append(label_name)
            image_name = label_name.split('label')[0]
            image_name = str(image_name) + 'image.tif'
            images_files.append(os.path.join(opts.file_dir, image_name))
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
    with open(r'./fed_gan/test_cls0_result_round.txt', 'a+', encoding='utf-8') as cls0:
        cls0.truncate(0)
    with open(r'./fed_gan/test_cls1_result_round.txt', 'a+', encoding='utf-8') as cls1:
        cls1.truncate(0)
    with open(r'./fed_gan/test_miou_result_round.txt', 'a+', encoding='utf-8') as miou:
        miou.truncate(0)
    with open(r'./fed_gan/test_pa_result_round.txt', 'a+', encoding='utf-8') as pa:
        pa.truncate(0)
    with open(r'./fed_gan/test_dice_result_round.txt', 'a+', encoding='utf-8') as dice:
        dice.truncate(0)
    epochs = 1
    for epoch in range(epochs, 1001):
        with torch.no_grad():
            model = model.eval()
            metrics.reset()

            miou = 0.0
            pa = 0.0
            dice = 0.0
            cls_iou = {}
            cls_iou[0] = 0.0
            cls_iou[1] = 0.0
            cls_iou[2] = 0.0
            i = 0

            opts.ckpt = f"/home/poac/7TB/yuanmingkang/deeplab_WHU2/deeplab_WHU2_train/fed_gan/checkpoints/{epoch}_deeplabv3plus_resnet101_ship_os16.pth"
            if opts.ckpt is not None and os.path.isfile(opts.ckpt):
                checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
                # Load the pretrained model
                model.load_state_dict(checkpoint["model_state"])
                # model = nn.DataParallel(model)
                model.to(device)
                print("Resume model from %s" % opts.ckpt)
                del checkpoint
            else:
                print("[!] Retrain")
                model = nn.DataParallel(model)
                model.to(device)

            for label_path,img_path,label_name1 in tqdm(zip(labels_files,images_files,labels_names)):
                i = i + 1

                img = Image.open(img_path).convert('RGB')
                label = Image.open(label_path).convert('RGB')

                label = encode_fn(label)

                resize = transforms.Resize([512, 512], Image.BICUBIC)
                label = Image.fromarray(label)

                label = resize(label)
                img = resize(img)
                label = np.array(label)
                img = np.array(img)

                img = transform(img).unsqueeze(0)  # To tensor of NCHW
                img = img.to(device)
                pred = model(img).max(1)[1].cpu().numpy()[0]  # HW

                metrics.update(label, pred)
                pred = decode_fn(pred)
                pred = np.uint8(pred)
                Image.fromarray(pred).save(f'/home/poac/7TB/yuanmingkang/deeplab_WHU2/deeplab_WHU2_test/fed_gan/seg_data/{label_name1}')
                score = metrics.get_results()
                miou = miou + float(score['Mean IoU'])
                pa = pa + float(score['Overall Acc'])
                dice = dice + float(score['Mean Dice'])
                class_iou = score['Class IoU']
                for k , v in class_iou.items():
                    cls_iou[k] = cls_iou[k]+ float(v)
        # message = f"-----------------epoch:{epoch}-----------------"
        # with open('./fed_gan/test_cls_result_round.txt', "a") as log_file:
        #     log_file.write('%s\n' % message)
        for i in range(0,opts.num_classes):
            # print(f"class {i}:  {cls_iou[i]/len(images_files)}")
            message = f"epoch:{epoch}, class{i}:{cls_iou[i]/len(images_files)}"
            with open(f'./fed_gan/test_cls{i}_result_round.txt', "a") as log_file:
                log_file.write('%s\n' % message)

        # print('miou:',miou/len(images_files))
        # print('pa:' , pa/len(images_files))
        # print('dice:' , dice/len(images_files))
        message = f"epoch:{epoch}, miou:{miou / len(images_files)}"
        with open('./fed_gan/test_miou_result_round.txt', "a") as log_file:
            log_file.write('%s\n' % message)

        message = f"epoch:{epoch}, pa:{pa/len(images_files)}"
        with open('./fed_gan/test_pa_result_round.txt', "a") as log_file:
            log_file.write('%s\n' % message)

        message = f"epoch:{epoch}, dice:{dice/len(images_files)}"
        with open('./fed_gan/test_dice_result_round.txt', "a") as log_file:
            log_file.write('%s\n' % message)
        print(f"epoch {epoch} is finished!")


if __name__ == '__main__':
    main()
