import torch
from torch import nn
import sys

from .base_model import BaseModel
from . import networks
import time
# from parse_config import ConfigParser
# import parse_config
# 三个地方要修改
class GeneratorModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input trainB to output trainB given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or val phase. You can use this flag to add training-specific or val-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='brats_split')
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
        # specify the training losses you want to print out. The training/val scripts will call <BaseModel.get_current_losses>
        # self.loss_names = ['G_GAN', 'G_L1', 'G_Seg', 'D_real', 'D_fake']
        self.loss_names = ['G_GAN_all', 'G_L1_all', 'G_perceptual_all']

        # self.visual_names = ['real_A_2', 'fake_B_2', 'real_B_2','real_A_7', 'fake_B_7', 'real_B_7']

        self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.G_learningrate, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <val>."""
        # recevie real_A
        # self.real_A = server.receive_object()
        # #
        # self.fake_B = (self.netG(self.real_A))
        #
        # server.send_object(self.fake_B)
        pass

    def optimize_parameters(self):

        return self.netG ,self.optimizer_G
        # while True:
            # self.forward()
            # receive real_A
        # n*d client_B A
        self.real_A = server.receive_object()
        # print(f'g receive real_a at device {self.real_A.device}')
        self.real_A.requires_grad_(True)
        # generate fake_B

        self.fake_B = (self.netG(self.real_A))
        # send fake_B
        # client_B Apwd

        server.send_object(self.fake_B)
        # set G's gradients to zero
        self.optimizer_G.zero_grad()
        # receive fake_B.grad
        self.fake_B_grad = server.receive_object()
        # print(f'g receive fake_b_grad at device {self.fake_B_grad.device}')
        # print(f'g at device {next(self.netG.parameters()).device}')

        # calculate graidents for G
        f = self.fake_B * self.fake_B_grad
        f = f.sum()
        f.backward()

        # udpate G's weights
        self.optimizer_G.step()

