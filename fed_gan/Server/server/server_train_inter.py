import matplotlib.pyplot as plt
import torch
from multi_thread_network.asyn_network import *
from multi_thread_network.network_base import *
import multiprocessing as mp
from time import *
from util.zip_unzip import *
from options.train_options import TrainOptions
from models import create_model
from data.ship_train_dataset import *
import numpy as np
import os
from  torchvision import utils
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import PySimpleGUI as sg
from  torchvision import utils
import shutil
import cv2
"""
    add Update_Frequency and D_learning
    modify client = [0,0,0,0,1,1,1,1]
    修改相对路径为绝对路径
    注释掉 amountdata相关代码
    加入提示弹窗
    195行附近有一个路径，需要修改
"""
def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        path1 = os.path.join(path,i)
        os.remove(path1)


def mkdir(path,flag):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print('new folder ok')
    else:
        if flag==0:
            del_file(path)
        print("folder existed !")

if __name__ == "__main__":
    # set model option
    opt = TrainOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    print(model)

    torch.backends.cudnn.benchmark = False
    # init parameters
    client_num = opt.client_num
    PORT = opt.port
    # with open(r'/home/poac/7TB/yuanmingkang/fed_gan_WHU2/loss/amount_data.txt', 'a+', encoding='utf-8') as amount_data:
    #     amount_data.truncate(0)
    # with open(r'/home/poac/4TB/yuanmingkang/fed_learning/gan_fed/GPU_server_client_sep/cityclassify_server/loss/loss_G.txt', 'a+', encoding='utf-8') as loss_G:
    #     loss_G.truncate(0)
    # start socket
    s = AsynServer(PORT, num_client=client_num, manager=mp.Manager())
    s.start()
    # re
    msg = {}
    msg['batch_size'] = opt.batch_size
    msg['D_learningrate'] = opt.D_learningrate
    msg['crop_size'] = opt.crop_size
    msg['load_size'] = opt.load_size
    msg['epoch'] = opt.epoch
    msg['Update_Frequency'] = opt.Update_Frequency
    print(f"msg{msg}")
    for key in range(client_num):
        s.asyn_send_object({key: msg})

    print("receive mask")
    #  receive mask
    objects = {key: None for key in range(client_num)}
    receive_objects = {key: None for key in range(client_num)}
    i = 0
    while len(receive_objects) != 0:
        s.asyn_receive_object(receive_objects)  # receive
        for key in receive_objects:
            if receive_objects[key] is None:
                continue
            mask_zip = receive_objects[key]
            # create ship_(i)
            path = f'../data/dataset/img_client_{i}'
            mkdir(path,0)# new empty folder

            path = f'../data/dataset/img_client_{i}/mask.zip'
            f = open(path, 'wb')
            f.write(mask_zip)
            f.close()
            # images mask
            unzip_mask_train_path = f'../data/dataset/train_{i}'
            mkdir(unzip_mask_train_path,0)
            unzip_file(path, unzip_mask_train_path)
            i += 1
        for key in list(receive_objects):
            if receive_objects[key] is not None:
                objects[key] = receive_objects[key]
                del receive_objects[key]
        if i >= client_num:
            break
    opt.dataroot='../data/dataset'
    test_mask_path = '../data/dataset/deeplab_train_img'
    sg.popup('Receive data successfully!', title='提示')
    Datasets = []
    masks_length = []
    mask_length_max = 0
    for i in range(client_num):
        Datasets.append(ShipTrainDataset(opt, i))
        temp = Datasets[i].__len__()
        masks_length.append(temp)
        print(temp)
        if temp>mask_length_max:
            mask_length_max = temp

    print(f"mask_len{masks_length}")
    print('training ...')
    print(f"batch_size{opt.batch_size}")
    for key in range(client_num):
        s.asyn_send_object({key: mask_length_max // opt.batch_size})

    total_item = 0

    # model.load_networks("new")
    # for epoch in range(100):
    start = False
    for epoch in range(int(opt.epoch)):
        step = 1
        #  get batchsize
        batch_size = int(opt.batch_size)
        #   get len
        length = mask_length_max
        indexs = [i for i in range(length)]
        np.random.shuffle(indexs)
        print("The num of epochs: " + str(epoch))
        client = []
        for i in range(opt.client_num):
            for j in range(opt.Update_Frequency):
                client.append(i)

        # else:
        #     netG, _ = model.optimize_parameters()
        #     model.save_networks('mean')
        for c in range(len(client)):
        # for j in range(client_num):
            j = client[c]
            training0 = {}
            training0['training'] = True
            training1 = {}
            training1['training'] = False

            if j == 0:
                s.asyn_send_object({0: training0})
                s.asyn_send_object({1: training1})
                if start:
                    model.load_networks('mean')
                for i in tqdm(range(length//batch_size)):
                    objects = {j: None}
                    receive_objects = {j: None}
                    netG0, optimizer_G0 = model.optimize_parameters()  # get netG
                    # print(netG0.state_dict())
                    fake_B_list = []
                    fake_B_dict = {}
                    real_A = [] # mask，mask对应的索引，
                    real_A_index=[] # mask对应的索引
                    real_A_transform_params = [] # 读取mask数据增强的参数
                    B_list = []
                    B_name = []
                    for k in range(batch_size):
                        temp, name_A = Datasets[j].__getitem__(indexs[(i*batch_size+k)%masks_length[j]])
                        real_A.append(temp['A'].cuda())
                        real_A_index.append(temp['A_index'])
                        B_list.append(temp['A_index'])
                        real_A_transform_params.append(temp['A_transform_params'])
                        B_name.append(name_A)
                    real_A = torch.stack(real_A, dim=0)
                    objects[j] = real_A
                    fake_B = netG0(real_A)
                    fake_B_dict[j] = fake_B
                    fake_txt = f'../data/dataset/fake_txt{j}'
                    shutil.rmtree(fake_txt)
                    os.mkdir(fake_txt)
                    if 1:
                    # if epoch > 30 and (epoch % 30) < 15:
                        for h in range(batch_size):
                            fake_name = ''.join(B_name[h].split('.')[:-1]) + '.png'

                            fake_path = f'../data/dataset/fake/{fake_name}'
                            fake_temp_path = f'{fake_txt}/{fake_name}'
                            tensor = fake_B[h]
                            # print(tensor)
                            fake_tensor = (tensor + 1) / 2
                            fake_tensor *= 255
                            fake_img = Image.fromarray(
                                fake_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')).convert(
                                'RGB')
                            fake_img.save(fake_path)
                            fake_img.save(fake_temp_path)
                    else:
                        for h in range(batch_size):
                            fake_name = ''.join(B_name[h].split('.')[:-1]) + '.png'
                            fake_path = f'../data/dataset/fake/{fake_name}'
                            tensor = fake_B[h]
                            # print(tensor)
                            fake_tensor = (tensor + 1) / 2
                            fake_tensor *= 255
                            fake_img = Image.fromarray(
                                fake_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')).convert(
                                'RGB')
                            fake_img.save(fake_path)

                            img = Image.open(fake_path)
                            img = np.array(img)
                            # imencode---
                            # img = cv2.imread(fake_path)
                            # print(np.array(img))
                            # print(np.array(img).shape)
                            params = [cv2.IMWRITE_JPEG_QUALITY, 70]
                            # params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                            _, img_encode = cv2.imencode('.jpg', img, params)
                            data_encode = np.array(img_encode)
                            str_encode = data_encode.tostring()
                            txt_name = ''.join(B_name[h].split('.')[:-1]) + '.txt'
                            with open(f'{fake_txt}/{txt_name}', 'wb') as f:
                                f.write(str_encode)
                                f.flush
                    zip_fake_txt = '/'.join(fake_txt.split('/')[:-1])
                    zip_fake_txt = os.path.join(zip_fake_txt, f'txt_zip{j}.zip')
                    zip_dir(fake_txt, zip_fake_txt)
                    fr = open(zip_fake_txt, 'rb')
                    temp_mask = fr.read()
                    fr.close()
                    result = [temp_mask, real_A_index, real_A_transform_params]

                    s.asyn_send_object({0: result})
                    s.asyn_send_object({1: training1})

                    grad_objects = {j: None}
                    receive_grad_objects = {j: None}
                    while len(receive_grad_objects) != 0:
                        send_fake_begin = time()
                        s.asyn_receive_object(receive_grad_objects)
                        if receive_grad_objects[j] is not None:
                            grad_objects[j] = receive_grad_objects[j]
                            del receive_grad_objects[j]
                        send_fake_end = time()

                    # 主进程进行计算loss
                    optimizer_G0.zero_grad()#清除以前的梯度
                    fake_B_grads_list = []
                    fake_B_grads_dict = {}

                    fake_B_grad = grad_objects[j].cuda()

                    fake_B_grads_dict[j] = fake_B_grad

                    # Note! fake_b_grad and fake_b should be aligned.

                    fake_B_list.append(fake_B_dict[j])
                    fake_B_grads_list.append(fake_B_grads_dict[j])
                    # fake_B_grads_tensor = torch.stack(fake_B_grads_list)
                    fake_B_grads_tensor = torch.stack(fake_B_grads_list).cuda()
                    fake_B_tensor = torch.stack(fake_B_list)

                    f = fake_B_tensor * fake_B_grads_tensor
                    f = f.sum()
                    f.backward()
                    # update weight
                    optimizer_G0.step()
                    total_item = total_item + 1

                model.save_networks(f'{epoch}_0')
                # torch.save(netG0, save_path_G0)
            if j == 1:
                s.asyn_send_object({0: training1})
                s.asyn_send_object({1: training0})
                if start:
                    model.load_networks('mean')
                for i in tqdm(range(length//batch_size)):
                    objects = {j: None}
                    receive_objects = {j: None}

                    netG1, optimizer_G1 = model.optimize_parameters()  # get netG
                    # print(netG1.state_dict())
                    fake_B_list = []
                    fake_B_dict = {}
                    real_A = [] # mask，mask对应的索引，
                    real_A_index=[] # mask对应的索引
                    real_A_transform_params = [] # 读取mask数据增强的参数
                    B_list = []
                    B_name = []
                    for k in range(batch_size):
                        temp, name_A = Datasets[j].__getitem__(indexs[(i*batch_size+k)%masks_length[j]])
                        real_A.append(temp['A'].cuda())
                        real_A_index.append(temp['A_index'])
                        B_list.append(temp['A_index'])
                        real_A_transform_params.append(temp['A_transform_params'])
                        B_name.append(name_A)
                    real_A = torch.stack(real_A, dim=0)
                    objects[j] = real_A
                    fake_B = netG1(real_A)
                    fake_B_dict[j] = fake_B
                    fake_txt = f'../data/dataset/fake_txt{j}'
                    shutil.rmtree(fake_txt)
                    os.mkdir(fake_txt)
                    if 1:
                        # if epoch > 30 and (epoch % 30) < 15:
                        for h in range(batch_size):
                            fake_name = ''.join(B_name[h].split('.')[:-1]) + '.png'

                            fake_path = f'../data/dataset/fake/{fake_name}'
                            fake_temp_path = f'{fake_txt}/{fake_name}'
                            tensor = fake_B[h]
                            # print(tensor)
                            fake_tensor = (tensor + 1) / 2
                            fake_tensor *= 255
                            fake_img = Image.fromarray(
                                fake_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')).convert(
                                'RGB')
                            fake_img.save(fake_path)
                            fake_img.save(fake_temp_path)
                    else:
                        for h in range(batch_size):
                            fake_name = ''.join(B_name[h].split('.')[:-1]) + '.png'
                            fake_path = f'../data/dataset/fake/{fake_name}'
                            tensor = fake_B[h]
                            # print(tensor)
                            fake_tensor = (tensor + 1) / 2
                            fake_tensor *= 255
                            fake_img = Image.fromarray(
                                fake_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')).convert(
                                'RGB')
                            fake_img.save(fake_path)

                            img = Image.open(fake_path)
                            img = np.array(img)
                            # imencode---
                            # img = cv2.imread(fake_path)
                            # print(np.array(img))
                            # print(np.array(img).shape)
                            params = [cv2.IMWRITE_JPEG_QUALITY, 70]
                            # params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
                            _, img_encode = cv2.imencode('.jpg', img, params)
                            data_encode = np.array(img_encode)
                            str_encode = data_encode.tostring()
                            txt_name = ''.join(B_name[h].split('.')[:-1]) + '.txt'
                            with open(f'{fake_txt}/{txt_name}', 'wb') as f:
                                f.write(str_encode)
                                f.flush
                    zip_fake_txt = '/'.join(fake_txt.split('/')[:-1])
                    zip_fake_txt = os.path.join(zip_fake_txt, f'txt_zip{j}.zip')
                    zip_dir(fake_txt, zip_fake_txt)
                    fr = open(zip_fake_txt, 'rb')
                    temp_mask = fr.read()
                    fr.close()
                    result = [temp_mask, real_A_index, real_A_transform_params]

                    s.asyn_send_object({0: training1})
                    s.asyn_send_object({1: result})

                    grad_objects = {j: None}
                    receive_grad_objects = {j: None}
                    while len(receive_grad_objects) != 0:
                        send_fake_begin = time()
                        s.asyn_receive_object(receive_grad_objects)
                        if receive_grad_objects[j] is not None:
                            grad_objects[j] = receive_grad_objects[j]
                            del receive_grad_objects[j]
                        send_fake_end = time()

                    # 主进程进行计算loss
                    optimizer_G1.zero_grad()#清除以前的梯度
                    fake_B_grads_list = []
                    fake_B_grads_dict = {}

                    fake_B_grad = grad_objects[j].cuda()

                    fake_B_grads_dict[j] = fake_B_grad

                    # Note! fake_b_grad and fake_b should be aligned.

                    fake_B_list.append(fake_B_dict[j])
                    fake_B_grads_list.append(fake_B_grads_dict[j])
                    # fake_B_grads_tensor = torch.stack(fake_B_grads_list)
                    fake_B_grads_tensor = torch.stack(fake_B_grads_list).cuda()
                    fake_B_tensor = torch.stack(fake_B_list)

                    f = fake_B_tensor * fake_B_grads_tensor
                    f = f.sum()
                    f.backward()
                    # update weight
                    optimizer_G1.step()
                    total_item = total_item + 1

                model.save_networks(f'{epoch}_1')
                # torch.save(netG1, save_path_G1)
        start = True
        if total_item % (2*(length // opt.batch_size)) == 0:
            model.update_learning_rate()
            # epoch = epoch + 1
        if epoch == opt.epoch:
            break
        save_path_G0 = f"../checkpoints/gan_city/{epoch}_0_net_G.pth"
        save_path_G1 = f"../checkpoints/gan_city/{epoch}_1_net_G.pth"
        # net_G = model.load_networks('mean')
        net_G = {}
        # net_G0 = model.load_networks(f'{epoch}_0')
        # net_G1 = model.load_networks(f'{epoch}_1')
        net_G0 = torch.load(save_path_G0)
        net_G1 = torch.load(save_path_G1)
        # net_G = netG.state_dict()
        # net_G0 = netG0.state_dict()
        # net_G1 = netG1.state_dict()
        for key in net_G0.keys():
            net_G[key] = (net_G0[key] + net_G1[key]) / float(2)
            # print(net_G[key])
            # print(net_G0[key])
            # print(net_G1[key])
        save_path_G = f"../checkpoints/gan_city/{epoch}_net_G.pth"
        save_path_G_mean = f"../checkpoints/gan_city/mean_net_G.pth"
        # model.save_networks(f'{epoch}')
        # model.save_networks('mean')
        torch.save(net_G, save_path_G)
        torch.save(net_G, save_path_G_mean)

        # message = f"---------------The amount of data in the {epoch} epochs---------------"
        # with open(
        #         '/home/poac/7TB/yuanmingkang/fed_gan_WHU2/loss/amount_data.txt',
        #         "a") as log_file:
        #     log_file.write('%s\n' % message)

    sg.popup('Start segmentation training data generation!', title='提示')
    shell_msg = f"python ./server_test.py --dataroot {test_mask_path} --load_size 512 --crop_size 512 --preprocess resize_and_crop &"
    os.system(shell_msg)
    sg.popup('Complete segmentation training data generation!', title='提示')
    s.close()
