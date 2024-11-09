import os

import torch
from torch import nn
import sys
from  torchvision import utils
print(sys.path)
# from models.UNet import setup_unet
from .perception_loss import vgg16_feat, perceptual_loss
from .base_model import BaseModel
from . import networks
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import time
# from parse_config import ConfigParser
"""
2022.8.23定义两个相对路径
LOSS_D_PATH 
LOSS_G_PATH
"""


class DiscriminatorEpochModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='resnet_9blocks', dataset_mode='brats')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--delta_perceptual', type=float, default=1.0, help='weight for perceptual loss')

            parser.add_argument('--lambda_G', type=float, default=0.1, help='weight for asyndgan G ')
            parser.add_argument('--lambda_D', type=float, default=0.05, help='weight for asyndgan D')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D','loss_G']
        self.loss_names = ['G_GAN', 'G_L1', 'G_perceptual', 'D_real', 'D_fake','D','G']
        # specify the trainB you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        #
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>

        # 定义G,D网络，根据D网络预设值的值来决定生成多少个实体
        if self.isTrain:
            self.model_names = ['D']

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output trainB; Therefore, #channels for D is input_nc + output_nc

            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # 定义优化器
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.D_learningrate, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D)

            self.vgg_model = vgg16_feat().cuda()
            # self.vgg_model = vgg16_feat().cpu()
            self.criterion_perceptual = perceptual_loss()
            # self.unet = setup_unet().cuda()
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap trainB in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'

        self.real_A = (input['A'].to(self.device))
        self.real_B = (input['B' ].to(self.device))
        self.image_paths = (input['A_paths'])
        # self.image_paths = (input['B_paths'])

    def forward(self):
    #     """Run forward pass; called by both functions <optimize_parameters> and <test>."""
    #     # 将mask传到G
    #     client_B.send_object(self.real_A)
    #     print("send real_A")
    #     print(type(self.real_A))
    #     # 收到G的fake_B
    #     self.fake_B = client_B.receive_object()
    #     print("accept fake_B")
    #     print(type(self.fake_B))
        pass


    # 第三步更新梯度
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_AB = torch.cat((self.real_A, self.fake_B),1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real)*self.opt.lambda_D #0.05
        loss_D_report = self.loss_D.item()
        loss_D_fake_report = self.loss_D_fake.item()
        loss_D_real_report = self.loss_D_real.item()
        message = f"loss_D,{loss_D_report},loss_D_fake,{loss_D_fake_report},loss_D_real,{loss_D_real_report}"
        LOSS_D_PATH = '../loss/loss_D.txt'
        with open(
                LOSS_D_PATH,
                "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        self.loss_D.backward()
        return loss_D_report, loss_D_fake_report, loss_D_real_report

    def optimize_parameters(self, client,dataset_real, batchsize):

        #  receive fake_B
        fake_result = client.receive_object()
        # fake_B_img = fake_result[0]

        self.fake_B = fake_result[0].cuda().detach()
        # self.fake_B = fake_result[0].detach()
        indexs = fake_result[1]

        l_A = []
        l_B = []
        name_B = []
        transform_params_A = fake_result[2]

        for i in range(len(indexs)):
            l1 = [indexs[i],transform_params_A[i]]
            temp = dataset_real.__getitem__(l1)
            l_A.append(temp['A'].to(self.device))
            l_B.append(temp['B'].to(self.device))
            name_B.append(temp['B_names'])
        for h in range(len(name_B)):
            fake_path = "../data/fakeimg"
            fake_path1 = fake_path + f'/{name_B[h]}'
            utils.save_image(self.fake_B[h], fake_path1, normalize=True)
        self.real_A = torch.stack(l_A,dim=0)

        self.real_A.requires_grad = True
        self.real_B = torch.stack(l_B, dim=0)
        self.real_B.requires_grad = True

        self.fake_B.requires_grad_(True)

        # upgrade D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()
        loss_D_report, loss_D_fake_report, loss_D_real_report = self.backward_D()

        self.optimizer_D.step()
        # if interval:
        #     self.set_requires_grad(self.netD, True)  # enable backprop for D
        #     self.optimizer_D.zero_grad()
        #     loss_D_report, loss_D_fake_report, loss_D_real_report = self.backward_D(epoch)
        #     self.optimizer_D.step()

        fake_AB = torch.cat((self.real_A, self.fake_B), 1)

        # 将patchGAN的结果返回给G，G和
        self.set_requires_grad(self.netD, False)
        pred_fake = self.netD(fake_AB)

        #
        self.loss_G_GAN = (self.criterionGAN(pred_fake, True))
        self.loss_G_L1 = (self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1)
        #
        pred_feat = self.vgg_model(self.fake_B)
        target_feat = self.vgg_model(self.real_B)
        self.loss_G_perceptual = (self.criterion_perceptual(pred_feat, target_feat) * self.opt.delta_perceptual)

        self.loss_G = (self.loss_G_GAN + self.loss_G_perceptual) * self.opt.lambda_G  # 0.1
        loss_G_report = self.loss_G.item()
        loss_G_GAN_report = self.loss_G_GAN.item()
        loss_G_perceptual_report = self.loss_G_perceptual.item()
        # 修改了保存格式(','分割)，以方便可视化。 每次新画图时就会清空文件。若需额外保存数据，可以备份。
        message = f"loss_G,{loss_G_report},loss_G_GAN,{loss_G_GAN_report},loss_G_perceptual,{loss_G_perceptual_report}"
        LOSS_G_PATH = '../loss/loss_G.txt'
        with open(
                LOSS_G_PATH,
                "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
        self.fake_B.retain_grad()
        # 根据每一项的梯度进行优化
        self.loss_G.backward()
        client.send_object(self.fake_B.grad.cpu())
        return loss_D_report, loss_D_fake_report, loss_D_real_report, loss_G_report, loss_G_GAN_report, loss_G_perceptual_report

        # if interval:
        #     return loss_D_report, loss_D_fake_report, loss_D_real_report, loss_G_report, loss_G_GAN_report, loss_G_perceptual_report
        # else:
        #     return loss_G_report, loss_G_GAN_report, loss_G_perceptual_report







