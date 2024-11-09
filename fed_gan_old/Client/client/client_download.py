from multi_thread_network.network_base import Client
import argparse
from util.zip_unzip import *
# 2022.8.24
# 增加了服务器未开启，模型下载成功的通知
import PySimpleGUI as sg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_ip', type=str)
    parser.add_argument('--port', type=str)

    args = parser.parse_args()
    server_ip = args.server_ip
    port = args.port
    try:
        client = Client(server_ip, int(port))
        client.start()
    except Exception as e:
        sg.popup_auto_close('Please click the [StartTrain] button on the server side !',title='提示',button_type=5,font='Any 18')
    else:
        sg.popup_auto_close('Start download model', title='提示', button_type=5, font='Any 18')
        msg = 'download'
        client.send_object(msg)
        deeplab_ckpt = client.receive_object()
        print("receive client")
        f = open("../checkpoints/deeplab/best_deeplabv3plus_resnet101_city_os16.pth", "wb+")
        f.write(deeplab_ckpt)
        f.close()
        sg.popup_auto_close('download successfully', title='提示', button_type=5, font='Any 18')
    # client.close()