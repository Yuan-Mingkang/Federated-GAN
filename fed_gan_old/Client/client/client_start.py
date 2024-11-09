import os

def start_gan_train(serverip,mask_path):
    port = 50012
    shell_msg = f"python ./client_gan_train_v2.py --server_ip {serverip} --port {port} --mask_path {mask_path}&"
    os.system(shell_msg)
    print(2)

if __name__ == '__main__':
    # serverip = input("server_ip:")
    # mask_path = input("mask_path:")
    serverip = '10.2.28.199'
    mask_path = '/home/poac/4TB/yuanmingkang/dataset/city/mask'
    start_gan_train(serverip, mask_path)