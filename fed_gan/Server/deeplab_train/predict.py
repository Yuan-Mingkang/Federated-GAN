# import network
# import utils
from deeplab import network
from deeplab import utils
import os
import argparse
import numpy as np
from deeplab.datasets import ShipSegmentation
from torchvision import transforms as T
# from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from PIL import Image

class Predict():
    def __init__(self):
        self.opts = self.get_argparser().parse_args()
        self.opts.ckpt = "../checkpoints/best_deeplabv3plus_resnet101_ship_os16.pth"
        # self.opts.save_val_results_to = "./results_1fake_new"

        self.opts.crop_size = 512
        self.opts.val_batch_size = 1
        self.opts.num_classes = 2
        self.opts.crop_val
        if self.opts.dataset.lower() == 'ship':
            self.decode_fn = ShipSegmentation.decode_target
            self.encode_fn = ShipSegmentation.encode_target

        # set gpus
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opts.gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Device: %s" % self.device)

        # Setup dataloader
        images_files = []
        labels_files = []

        labels_name = os.listdir(self.opts.file_dir)
        for label_name in labels_name:
            if label_name.endswith('.png'):
                labels_files.append(os.path.join(self.opts.file_dir, label_name))
                image_name = label_name.split('.')[:-1]
                image_name = str(image_name[0]) + '.png'
                images_files.append(os.path.join(self.opts.file_dir, image_name))
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

        self.model = model_map[self.opts.model](num_classes=self.opts.num_classes, output_stride=self.opts.output_stride)
        if self.opts.separable_conv and 'plus' in self.opts.model:
            network.convert_to_separable_conv(self.model.classifier)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)

        if self.opts.ckpt is not None and os.path.isfile(self.opts.ckpt):
            # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
            checkpoint = torch.load(self.opts.ckpt, map_location=torch.device('cpu'))
            # Load the pretrained model
            self.model.load_state_dict(checkpoint["model_state"])
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)
            print("Resume model from %s" % self.opts.ckpt)
            del checkpoint
        else:
            print("[!] Retrain")
            self.model = nn.DataParallel(self.model)
            self.model.to(self.device)

        if self.opts.crop_val:
            self.transform = T.Compose([
                # T.Resize(opts.crop_size),
                # T.CenterCrop(opts.crop_size),
                T.Resize(self.opts.crop_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        if self.opts.save_val_results_to is not None:
            os.makedirs(self.opts.save_val_results_to, exist_ok=True)

    def get_argparser(self):
        parser = argparse.ArgumentParser()

        # Datset Options
        # parser.add_argument("--input", type=str, required=True,
        #                     help="path to a single images_beifen or images_beifen directory")
        parser.add_argument("--file_dir", type=str,
                            help="path to a single images_beifen or images_beifen directory")
        parser.add_argument("--num_classes", type=int,
                            help="path to a single images_beifen or images_beifen directory")
        parser.add_argument("--dataset", type=str, default='ship',
                            choices=['ship', 'cityscapes'], help='Name of training set')
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
        parser.add_argument("--gpu_id", type=str, default='0',
                            help="GPU ID")
        return parser

    def start(self,img_path,img_name):
        with torch.no_grad():
            model = self.model.eval()

            img = Image.open(img_path).convert('RGB')

            img = self.transform(img).unsqueeze(0)  # To tensor of NCHW
            img = img.to(self.device)
            # Visualize segmentation outputs:
            pred = model(img).max(1)[1].cpu().numpy()[0]  # HW


            pred = self.decode_fn(pred)
            pred_save = Image.fromarray(np.uint8(pred))

            if self.opts.save_val_results_to:
                pred_save.save(os.path.join(self.opts.save_val_results_to, img_name + '.png'))




