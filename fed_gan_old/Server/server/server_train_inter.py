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
    # message = "---------------Amount of data before training starts---------------"
    # with open('/home/poac/7TB/yuanmingkang/fed_gan_WHU2/loss/amount_data.txt',
    #           "a") as log_file:
    #     log_file.write('%s\n' % message)

    # for epoch in range(100):
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


        for k in range(len(client)):
        # for j in range(client_num):
            j = client[k]
            training0 = {}
            training0['training'] = True
            training1 = {}
            training1['training'] = False
            if j == 0:
                s.asyn_send_object({0: training0})
                s.asyn_send_object({1: training1})
            if j == 1:
                s.asyn_send_object({0: training1})
                s.asyn_send_object({1: training0})
            for i in tqdm(range(length//batch_size)):

                objects = {j: None}
                receive_objects = {j: None}
                netG, optimizer_G = model.optimize_parameters()  # get netG

                fake_B_list = []
                fake_B_dict = {}
                real_A = [] # mask，mask对应的索引，
                real_A_index=[] # mask对应的索引
                real_A_transform_params = [] # 读取mask数据增强的参数


                for k in range(batch_size):
                    temp = Datasets[j].__getitem__(indexs[(i*batch_size+k)%masks_length[j]])
                    real_A.append(temp['A'].cuda())
                    real_A_index.append(temp['A_index'])

                    real_A_transform_params.append(temp['A_transform_params'])

                real_A = torch.stack(real_A, dim=0)
                objects[j] = real_A
                fake_B = netG(real_A)
                fake_B_dict[j] = fake_B
                temp_fake_B = fake_B.cpu()
                temp_fake_B = temp_fake_B.detach()
                result = [temp_fake_B, real_A_index,real_A_transform_params]
                if j == 0:
                    s.asyn_send_object({0: result})
                    s.asyn_send_object({1: training1})
                if j == 1:
                    s.asyn_send_object({0: training1})
                    s.asyn_send_object({1: result})
                # for h in range(batch_size):
                #     # ????
                #     fake_path = f'/home/poac/7TB/yuanmingkang/fed_gan_WHU2/data/dataset/fake_B_{j}/{B_name[h]}'
                #     utils.save_image(fake_B[h], fake_path, normalize=True)
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
                optimizer_G.zero_grad()#清除以前的梯度
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
                optimizer_G.step()
                total_item = total_item + 1


                message = f"---------------The amount of data in the {i} batchs---------------"
                # with open(
                #         '/home/poac/7TB/yuanmingkang/fed_gan_WHU2/loss/amount_data.txt',
                #         "a") as log_file:
                #     log_file.write('%s\n' % message)
        if total_item % (2*(length // opt.batch_size)) == 0:
            model.update_learning_rate()
            # epoch = epoch + 1

        if epoch == opt.epoch:
            break
        model.save_networks('latest')
        model.save_networks(int(epoch))
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
