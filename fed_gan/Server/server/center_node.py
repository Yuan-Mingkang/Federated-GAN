import torch
from tqdm import tqdm
from multi_thread_network.asyn_network import *
from multi_thread_network.network_base import *
import multiprocessing as mp

client_num = 2
port = 50015
s = AsynServer(port, num_client=client_num, manager=mp.Manager())
s.start()

epochs = 200
batch_num = 160
# with open(r'/home/poac/4TB/yuanmingkang/center_gan/amount_data.txt',
#           'a+', encoding='utf-8') as amount_data:
#     amount_data.truncate(0)
save_path_now = f"/home/poac/4TB/yuanmingkang/center_gan/checkpoints/now_net.pth"
for epoch in range(52, epochs):
    print("The current epoch is:" + str(epoch))
    for batch in tqdm(range(batch_num)):
        pth = {key: None for key in range(client_num)}
        receive_objects = {key: None for key in range(client_num)}
        i = 0
        while len(receive_objects) != 0:
            s.asyn_receive_object(receive_objects)
            for key in receive_objects:
                if receive_objects[key] is None:
                    continue
                i += 1
            for key in list(receive_objects):
                if receive_objects[key] is not None:
                    pth[key] = receive_objects[key]
                    del receive_objects[key]
            if i >= client_num:
                break
        net_G = {}
        net_D = {}

        for key in pth[0][0].keys():
            net_G[key] = (pth[0][0][key] + pth[1][0][key]) / 2
        for key in pth[0][1].keys():
            net_D[key] = (pth[0][1][key] + pth[1][1][key]) / 2
        net = {0: net_G, 1: net_D}
        for key in range(client_num):
            s.asyn_send_object({key: net})
    save_path_G = f"/home/poac/4TB/yuanmingkang/center_gan/checkpoints/{epoch}_net_G.pth"
    save_path_D = f"/home/poac/4TB/yuanmingkang/center_gan/checkpoints/{epoch}_net_D.pth"
    torch.save(net_G, save_path_G)
    torch.save(net_D, save_path_D)

