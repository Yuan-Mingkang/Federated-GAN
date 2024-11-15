from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import ShipSegmentation
from utils import ext_transforms as et
from metrics import StreamSegMetrics

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from  torchvision import utils as utl
"""
train_script
    python main_city.py --data_root /home/ilab/gxy/images/global_city/train_val --dataset city --num_classes 3 --batch_size 12 --crop_size 512 --vis_port 8097 --continue_training --ckpt /home/ilab/gxy/deeplab/checkpoints_city/latest_deeplabv3plus_resnet101_city_os16.pth  
"""

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='',
                        help="path to Dataset")
    parser.add_argument("--mask_path", type=str, default='/home/poac/4TB/yuanmingkang/dataset/WHU_Building/Satellite dataset вё (global cities)/train/mask_aug',
                        help="path to mask")
    parser.add_argument("--train_path", type=str, default='/home/poac/7TB/yuanmingkang/gan_lr_WHU2/dataset/fake_B',
                        help="path to gan_lr_fake")
    parser.add_argument("--dataset", type=str, default='ship',
                        choices=['voc', 'cityscapes', 'dota', 'ship'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=['deeplabv3_resnet50', 'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default= True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=3000e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--total_epochs", type=int, default=1000,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=True,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=10,
                        help='batch size (default: 8)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=200,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")


    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=True,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser


def get_dataset(opts):
    """ Dataset And Augmentation
    """

    train_transform = et.ExtCompose([
        # modify ext_transforms.py
        et.ExtScale(opts.crop_size),
        # et.ExtResize(size=opts.crop_size),
        # et.ExtRandomScale((0.5, 2.0)),
        # et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
        # et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    if opts.crop_val:
        val_transform = et.ExtCompose([
            # et.ExtResize(opts.crop_size),
            # et.ExtCenterCrop(opts.crop_size),
            et.ExtScale(opts.crop_size),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
    else:
        val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    train_dst = ShipSegmentation(mask_path=opts.mask_path, train_path=opts.train_path, image_set='train', transform=train_transform)
    # val_dst = CitySegmentation(mask_path=opts.mask_path, train_path=opts.train_path, image_set='val', transform=val_transform)

    return train_dst
    # return train_dst, val_dst


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()

            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    # print(type(target))
                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)
                    # target =target.astype(np.uint8)
                    # pred = pred.astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    opts = get_argparser().parse_args()
    #
    # opts.data_root ='/home/ilab/gxy/images/train_val'
    opts.dataset = 'ship'
    opts.num_classes = 2
    opts.batch_size =10
    opts.crop_size =512
    opts.vis_port = 8097


    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'ship' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst = get_dataset(opts)
    # train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=0,drop_last=True)
    # val_loader = data.DataLoader(
    #     val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2,drop_last=True)
    # print("Dataset: %s, Train set: %d, Val set: %d" %
    #       (opts.dataset, len(train_dst), len(val_dst)))
    print("Dataset: %s, Train set: %d" %
          (opts.dataset, len(train_dst)))

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

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    def save_ckpt(path):
        """ save current model
        """
        # utils.mkdir(path)
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    # utils.mkdir('checkpoints_city')
    # Restore
    best_score = 0.0
    cur_epochs = 0
    # opts.ckpt = "/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab_ship_train/real/checkpoints/414_deeplabv3plus_resnet101_city_os16.pth"
    # opts.ckpt = "/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab_ship_train/gan_lr/checkpoints/deeplab_city/375_deeplabv3plus_resnet101_city_os16.pth"
    # opts.ckpt = "/home/poac/4TB/yuanmingkang/deeplab/deeplab_city/deeplab_ship_train/fed_gan/checkpoints/deeplab_city/375_deeplabv3plus_resnet101_city_os16.pth"
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    # vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
    #                                   np.int32) if opts.enable_vis else None  # sample idxs for visualization
    # denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images_beifen
    #
    # if opts.test_only:
    #     model.eval()
    #     val_score, ret_samples = validate(
    #         opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
    #     print(metrics.to_str(val_score))
    #     return

    interval_loss = 0
    epoch_loss = 0
    a = 0
    with open(r'./gan_lr/loss/loss_deeplab.txt', 'a+', encoding='utf-8') as loss_deeplab:
        loss_deeplab.truncate(0)
    with open(r'./gan_lr/loss/epoch_loss_deeplab.txt', 'a+', encoding='utf-8') as epoch_loss_deeplab:
        epoch_loss_deeplab.truncate(0)
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        print("The current epoch is :" + str(cur_epochs))
        cur_itrs = 0
        for images, labels, names in tqdm(train_loader):

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            preds_train = outputs.detach().max(dim=1)[1].cpu().numpy()
            # targets_train = labels.cpu().numpy()
            # print(outputs.size())
            # print(labels.size())

            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            for h in range(opts.batch_size):
                pred_train = preds_train[h]
                pred_train = train_loader.dataset.decode_target(pred_train).astype(np.uint8)
                Image.fromarray(pred_train).save(f'/home/poac/7TB/yuanmingkang/deeplab_WHU2/deeplab_WHU2_train/gan_lr/seg_data/{names[cur_itrs*opts.batch_size+h][1]}')


            # if vis is not None:
            #     vis.vis_scalar('Loss', cur_itrs, np_loss)
            epoch_loss += np_loss
            cur_itrs += 1
            a += 1
            # if (cur_itrs) % 10 == 0:
            #     interval_loss = interval_loss / 10
            #     # print("Epoch %d, Itrs %d/%d, Loss=%f" %
            #     #       (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
            #     message = f"epoch:{cur_epochs},cur_iters:{cur_itrs},loss:{interval_loss:.5f}"
            #
            #     # print(message)  # print the message
            #     with open('./loss/loss_deeplab.txt', "a") as log_file:
            #         log_file.write('%s\n' % message)  # save the message
            #
            #     interval_loss = 0.0
            message = f"num:{a},epoch:{cur_epochs},cur_iters:{cur_itrs},loss:{interval_loss:.5f}"


            with open('./gan_lr/loss/loss_deeplab.txt', "a") as log_file:
                log_file.write('%s\n' % message)


            # if (cur_itrs) % opts.val_interval == 0:
            #     save_ckpt('./checkpoints/deeplab_city/latest_%s_%s_os%d.pth' %
            #               (opts.model, opts.dataset, opts.output_stride))
            #     print("validation...")
            #     model.eval()
            #     val_score, ret_samples = validate(
            #         opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
            #         ret_samples_ids=vis_sample_id)
            #     print(metrics.to_str(val_score))
            #     if val_score['Mean IoU'] > best_score:  # save best model
            #         best_score = val_score['Mean IoU']
            #         save_ckpt('./checkpoints/deeplab_city/best_%s_%s_os%d.pth' %
            #                   (opts.model, opts.dataset, opts.output_stride))
            #     model.train()
            scheduler.step()

        save_ckpt(f'./gan_lr/checkpoints/{cur_epochs}_%s_%s_os%d.pth' %
                  (opts.model, opts.dataset, opts.output_stride))
        save_ckpt('./gan_lr/checkpoints/latest_%s_%s_os%d.pth' %
                  (opts.model, opts.dataset, opts.output_stride))

        epoch_loss = epoch_loss / cur_itrs
        print("Epoch %d,  Loss=%f" %
              (cur_epochs, epoch_loss))
        message = f"epoch:{cur_epochs}, epoch_loss:{epoch_loss:.5f}"

        with open('./gan_lr/loss/epoch_loss_deeplab.txt', "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


        epoch_loss = 0.0


        save_ckpt('/home/poac/7TB/yuanmingkang/deeplab_WHU2/deeplab_WHU2_train/gan_lr/checkpoints/%d_%s_%s_os%d.pth' %
            (cur_epochs, opts.model, opts.dataset, opts.output_stride))

        if cur_epochs >= opts.total_epochs:
            return
if __name__ == '__main__':
     main()
# 35045
# 2837