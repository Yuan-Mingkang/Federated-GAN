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
import PySimpleGUI as sg
from  torchvision import utils
import shutil
from tqdm import tqdm
import torchvision
import torchvision.transforms as transformers
import cv2

"""
复制了最原始的版本，绝对路径相对麻烦
增加了D的学习率
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
    torch.backends.cudnn.benchmark = False
    # init parameters
    client_num = opt.client_num
    PORT = opt.port
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
    sg.popup_auto_close('Start receiving data', title='提示', button_type=5, font='Any 18')
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

            # delete ship_i
            # test_mask
            unzip_mask_test_path = f'../data/dataset/mask'
            mkdir(unzip_mask_test_path,i)
            unzip_file(path, unzip_mask_test_path)

            unzip_deeplab_train_img_path = f'../data/dataset/deeplab_train_img'
            mkdir(unzip_deeplab_train_img_path, i)
            unzip_file(path, unzip_deeplab_train_img_path)

            i += 1
        for key in list(receive_objects):
            if receive_objects[key] is not None:
                objects[key] = receive_objects[key]
                del receive_objects[key]
        if i >= client_num:
            break
    opt.dataroot='../data/dataset'
    test_mask_path = '../data/dataset/deeplab_train_img'
    s.close()
    sg.popup_auto_close('Receive data successfully!', title='提示', button_type=5, font='Any 18')

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
    for key in range(client_num):
        s.asyn_send_object({key: mask_length_max // opt.batch_size})
    total_item = 0
    for epoch in range(int(opt.epoch)):
        step = 1

        #  get batchsize
        batch_size = int(opt.batch_size)
        #   get len
        length = mask_length_max

        indexs = [i for i in range(length)]
        np.random.shuffle(indexs)


        for i in tqdm(range(length//batch_size)):

            objects = {key: None for key in range(client_num)}
            receive_objects = {key: None for key in range(client_num)}
            netG, optimizer_G = model.optimize_parameters()  # get netG

            fake_B_list = []
            fake_B_dict = {}

            for j in range(client_num):

                real_A = [] # mask，mask对应的索引，
                real_A_index=[] # mask对应的索引
                real_A_transform_params = [] # 读取mask数据增强的参数
                B_list = []
                B_name = []
                for k in range(batch_size):
                    temp, name_A = Datasets[j].__getitem__(indexs[(i * batch_size + k) % masks_length[j]])
                    real_A.append(temp['A'].cuda())
                    real_A_index.append(temp['A_index'])
                    B_list.append(temp['A_index'])
                    real_A_transform_params.append(temp['A_transform_params'])
                    B_name.append(name_A)

                real_A = torch.stack(real_A, dim=0)
                objects[j] = real_A

                fake_B = netG(real_A)
                fake_img = fake_B[0]

                fake_B_dict[j] = fake_B
                #----trans-txt----
                fake_txt = f'../data/dataset/fake_txt{j}'
                shutil.rmtree(fake_txt)
                os.mkdir(fake_txt)
                if epoch == 400:
                    for h in range(batch_size):
                        fake_name = ''.join(B_name[h].split('.')[:-1]) + '.png'

                        fake_path = f'../data/dataset/fake/{fake_name}'
                        fake_temp_path = f'{fake_txt}/{fake_name}'
                        tensor = fake_B[h]
                        # print(tensor)
                        fake_tensor = (tensor + 1) / 2
                        fake_tensor *= 255
                        fake_img = Image.fromarray(fake_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')).convert('RGB')
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
                # ----trans-jpg----
                #temp_fake_B = fake_B.cpu()
                result = [temp_mask, real_A_index,real_A_transform_params]

                #result = [temp_fake_B, real_A_index,real_A_transform_params]
                s.asyn_send_object({j: result})

            grad_objects = {key: None for key in range(client_num)}
            receive_grad_objects = {key: None for key in range(client_num)}
            while len(receive_grad_objects) != 0:
                send_fake_begin = time()
                s.asyn_receive_object(receive_grad_objects)
                for key in list(receive_grad_objects):
                    if receive_grad_objects[key] is not None:
                        grad_objects[key] = receive_grad_objects[key]
                        del receive_grad_objects[key]
                send_fake_end = time()

            # 主进程进行计算loss
            optimizer_G.zero_grad()#清除以前的梯度
            fake_B_grads_list = []
            fake_B_grads_dict = {}

            for key in grad_objects:
                fake_B_grad = grad_objects[key].cuda()
                fake_B_grads_dict[key] = fake_B_grad

            # Note! fake_b_grad and fake_b should be aligned.
            for key in fake_B_dict:
                fake_B_list.append(fake_B_dict[key])
                fake_B_grads_list.append(fake_B_grads_dict[key])

            fake_B_grads_tensor = torch.stack(fake_B_grads_list).cuda()
            fake_B_tensor = torch.stack(fake_B_list)

            f = fake_B_tensor * fake_B_grads_tensor
            f = f.sum()
            message = f"(total_items:,{total_item},loss,{f:.2f},)"

            with open('../loss/loss_gan.txt', "a") as log_file:
                log_file.write('%s\n' % message)  # save the message

            f.backward()
            # update weight
            optimizer_G.step()
            total_item = total_item + 1


            if total_item % (length // opt.batch_size) == 0:
                model.update_learning_rate()
                epoch = epoch + 1
                print(f"current{epoch}")
            if epoch % 5 == 0:
                model.save_networks(int(epoch))
            # if total_item % 1000 == 0:
                # model.save_networks('latest')
                # model.save_networks(int(total_item))
            if epoch == opt.epoch:
                break
    sg.popup_auto_close('Start segmentation training data generation!', title='提示', button_type=5, font='Any 18')
    # 将生成出来的图片保存在输入图片的相同目录下 test_mask_path中
    shell_msg = f"python ./server_test.py --dataroot {test_mask_path} --load_size 512 --crop_size 512 --preprocess resize_and_crop &"
    os.system(shell_msg)
    sg.popup_auto_close('Complete segmentation training data generation!', title='提示', button_type=5, font='Any 18')
    s.close()
