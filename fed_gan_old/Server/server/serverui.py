# -*- coding: UTF-8 -*-
import os, signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
"""
2022.08.03
strain_gan_train_by_epoch(msg)
add learningrate; Update Frequency(当客户端数据分布相差较大时，使用epoch为单位的频率更新模型)
"""
""""
启动各项服务的脚本
"""
def killport(port):
    for line in os.popen("ps ax | grep " + str(port) + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)
    print("kill successful")

# def start_train_all():
#     print(f"start gan train all")
#     # folder contain images
#     train_path = '../data/city/images'
#     shell_msg_train = f'python ../gan_all/train.py --dataroot {train_path} --name gan_city_all --model asyndgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode city_split --pool_size 0 ' \
#                 '--gpu_ids 0 --batch_size 1 --num_threads 1 --preprocess scale_width --load_size 512 --crop_size 512 --epoch 100'
#     os.system(shell_msg_train)


# def start_deeplab_train_all():
#     print("start_deeplab_train_all")
#     train_deeplab_path = '../data/city/train'
#     shell_msg = f'python ../deeplab/main_city.py --data_root {train_deeplab_path} --dataset city --num_classes 3 --batch_size 12 --crop_size 512 --gan_class deeplab_city_all --total_itrs 10600 --vis_port 8097'
#     os.system(shell_msg)

def strain_gan_train_by_epoch(msg):
    port = 50012
    batchsize = msg['batch_size']
    load_size = msg['load_size']
    crop_size = msg['crop_size']
    epoch = int(msg['epoch'])
    client_num = msg['client_num']
    G_learningrate = msg['G_learningrate']
    D_learningrate = msg['D_learningrate']
    Update_Frequency = msg['Update_Frequency']

    shell_msg = f"python ./server_train_inter.py --model generator --Update_Frequency {Update_Frequency} --D_learningrate {D_learningrate} --G_learningrate {G_learningrate} --client_num {client_num} --epoch {epoch} --batch_size {batchsize} --load_size {load_size} --crop_size {crop_size} --port {port}  --preprocess resize_and_crop --dataset_mode ship_train &"
    print(f"mask shell_msg {shell_msg}")
    os.system(shell_msg)
#  gan训练脚本
def start_gan_train(msg):
    port = 50012
    batchsize = msg['batch_size']
    load_size = msg['load_size']
    crop_size = msg['crop_size']
    epoch = int(msg['epoch'])
    client_num = msg['client_num']
    G_learningrate = msg['G_learningrate']
    D_learningrate = msg['D_learningrate']
    Update_Frequency = msg['Update_Frequency']

    shell_msg = f"python ./server_train.py --model generator --Update_Frequency {Update_Frequency} --D_learningrate {D_learningrate} --G_learningrate {G_learningrate} --client_num {client_num} --epoch {epoch} --batch_size {batchsize} --load_size {load_size} --crop_size {crop_size} --port {port}  --preprocess resize_and_crop --dataset_mode ship_train &"
    print(f"gan_train shell_msg {shell_msg}")
    os.system(shell_msg)


# 生成假图片的脚本
def generate_fake(client_num):
    port = 50013
    shell_msg = f"python ./generate_fake.py --port {port} --client_num {client_num}&"
    os.system(shell_msg)


# deeplab训练脚本
def start_deeplab():
    # 修改 --dataset --numclasses 切换数据集 main_ship.py 删除--gan_city
    data_root = '../data/dataset/deeplab_train_img'
    shell_msg = f'python ../deeplab_train/main_ship.py --data_root {data_root} --dataset ship --num_classes 2 --batch_size 4 --crop_size 512 --vis_port 8097 --total_itrs 10600&'
    os.system(shell_msg)
#  下载模型脚本
def download(client_num):
    port = 50014
    shell_msg = f"python ./server_download.py --port {port} --client_num {client_num}&"
    os.system(shell_msg)

#
def kill_train(name):
    for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        print(pid)
        os.kill(int(pid), signal.SIGKILL)
    print("Process Successfully terminated")
"""各项服务脚本结束"""


"""css"""
# 文本大小
TEXT_SIZE = (12, 1)
# 按钮大小
BUTTON_SIZE = (15, 1)
# 输入文本大小
INPUT_TEXT_SIZE = (8, 20)
# 窗口大小
WINDOWS_SIZE = (1366, 768)
# 展示loss界面大小
FIGURE_SIZE = (800,494)
# GAN模大小
GAN_FRAME_SIZE = (200, 350)
# 文字样式及大小
FONT_H1_SIZE = ('宋体',12)
FONT_H2_SIZE = 'Any 12'
FONT_BUTTON_SIZE =  ('宋体',10)
FONT_TEXT_SIZE = 'Any 18'
"""css"""

"""UI start"""
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


server_start = [

    # [sg.Text('serverport', size=(10, 0)), sg.InputText(size=(12, 20), key="PORT", default_text='50012')],
        [sg.Text('clientnum', size=TEXT_SIZE,font=FONT_TEXT_SIZE), sg.InputText(size=INPUT_TEXT_SIZE, key="CLIENTNUM", default_text=2)],
    [sg.Button('StartServer', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=("STARTSERVER"))],
]
gan_train = [

    [sg.Text('batchsize', size=TEXT_SIZE,font=FONT_TEXT_SIZE), sg.InputText(size=INPUT_TEXT_SIZE, key="BATCHSIZE", default_text=5)],
    [sg.Text('G_learningrate', size=TEXT_SIZE,font=FONT_TEXT_SIZE), sg.InputText(size=INPUT_TEXT_SIZE, key="G_LEARNINGRATE", default_text=0.0002)],
    [sg.Text('D_learningrate', size=TEXT_SIZE, font=FONT_TEXT_SIZE),sg.InputText(size=INPUT_TEXT_SIZE, key="D_LEARNINGRATE", default_text=0.0001)],
    [sg.Text('imgsize', size=TEXT_SIZE,font=FONT_TEXT_SIZE), sg.InputText(size=INPUT_TEXT_SIZE, key="IMGSIZE", default_text=512)],
    [sg.Text('epoch', size=TEXT_SIZE,font=FONT_TEXT_SIZE), sg.InputText(size=INPUT_TEXT_SIZE, key="EPOCH", default_text=100)],
    [sg.Text('Update_Frequency', size=TEXT_SIZE, font=FONT_TEXT_SIZE),
     sg.InputText(size=INPUT_TEXT_SIZE, key="UPDATE_FREQUENCY", default_text=0)],

    # [sg.Button('sendParameters', size=(22, 1), key=("SENDPARAMETER"))],
    [sg.Button('StartTrain', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=("STARTTRAIN"))],
    [sg.Button('StopTrain', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=("STOPSTRAIN1"))],
]
train_all = [
    [sg.Button('StartTrainAll', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=("STARTTRAINALL"))],
]
draw_pic = [
    # [sg.Text('loss', font='Any 18')],
    [sg.Canvas(key='canvas',size=FIGURE_SIZE,background_color='white')]
]

deeplab_train = [
    [sg.Button('StartTrain', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE,key=("DEEPLABTRAIN"))],
    [sg.Button('StopTrain', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=("STOPSTRAIN2"))],
    # [sg.Button('集中训练', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=("DEEPLABTRAINALL"))],
]
# deeplab_download = [
#     [sg.Button('Download', size=BUTTON_SIZE, key=("DOWNLOAD"))],
# ]
layout = [
    [sg.Frame('ServerInit', server_start, title_color='black', font=FONT_H1_SIZE)],
    # [sg.Frame('集中训练', train_all, title_color='black', font=FONT_H1_SIZE)],
    [sg.Frame('GanTrain', gan_train, title_color='black', font=FONT_H1_SIZE,),
     sg.Frame('Loss', draw_pic, title_color='black',background_color='white',pad=((200,0),(0,0)),size=FIGURE_SIZE, font=FONT_H1_SIZE)],
    [sg.Frame('DeepLabTrain', deeplab_train, title_color='black', font=FONT_H1_SIZE)],

]

window = sg.Window('Server', layout, size=WINDOWS_SIZE, resizable=True,location=(0,0)).finalize()
plt.figure(1)
fig = plt.gcf()
DPI = fig.get_dpi()
fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)
"""ui end"""

serverop = None
mask_path = None
fake_image_path = None
index = 0
first_delete_gan = 1
first_delete_deeplab = 1
client_num = 1
loss_gan_path = '../loss/loss_gan.txt'
loss_deeplab_path = '../loss/loss_deeplab.txt'

while True:
    if index == 1:
        if first_delete_gan == 1:
            if os.path.exists(loss_gan_path):
                os.remove(loss_gan_path)
            os.system(f'touch {loss_gan_path}')
            first_delete_gan = 0
        with open(loss_gan_path, "r") as log_file:
            content = log_file.readlines()
            iters = []
            loss = []
            for line in content:
                line = line[:]
                list_l = line.split(',')
                iters.append(int(list_l[1]))
                loss.append(float(list_l[3]))
        # plt.cla()
        plt.plot(iters, loss,color='red')
        fig_canvas_agg.draw()

    elif index == 2:

        if first_delete_deeplab == 1:
            # plt.cla()
            # plt.clf()
            if os.path.exists(loss_deeplab_path):
                os.remove(loss_deeplab_path)
            os.system(f'touch {loss_deeplab_path}')
            first_delete_deeplab = 0
        with open(loss_deeplab_path, "r") as log_file:
            content = log_file.readlines()
            iters = []
            loss = []
            for line in content:
                line = line[:]
                list_l = line.split(',')
                iters.append(int(list_l[3]))
                loss.append(float(list_l[5]))
        # plt.cla()
        plt.clf()
        plt.plot(iters, loss,color='red')
        fig_canvas_agg.draw()

    event, values = window.read(timeout=500)
    if event == None:
        break
    # if event == 'STARTTRAINALL':
    #     start_train_all()
    if event == 'STARTSERVER':
        client_num = values['CLIENTNUM']
        # 清除端口占用
        # kill_train('50012')
        # kill_train('50013')
        # kill_train('50014')
        # 开启生成假图片模块
        generate_fake(client_num)
        # 开启模型下载服务
        download(client_num)
        sg.popup_auto_close("Start Server!", title='提示', button_type=5, font='Any 18')
        # sg.popup('all client connect')
    #  start images
    if event == 'STARTTRAIN':
        msg = {}
        batchsize = values['BATCHSIZE']
        G_learningrate = values['G_LEARNINGRATE']
        D_learningrate = values['D_LEARNINGRATE']
        imgsize = values['IMGSIZE']
        epoch = values['EPOCH']
        Update_Frequency = values['UPDATE_FREQUENCY']
        msg['batch_size'] = batchsize
        msg['G_learningrate'] = G_learningrate
        msg['D_learningrate'] = D_learningrate
        msg['load_size'] = 512
        msg['crop_size'] = imgsize
        msg['epoch'] = epoch
        msg['client_num'] = client_num
        msg['Update_Frequency'] = Update_Frequency

        if Update_Frequency == '0':
            start_gan_train(msg)
        else:
            strain_gan_train_by_epoch(msg)
        index = 1

    # stop strain
    if event == 'STOPSTRAIN1':
        name = "server_train.py"
        kill_train(name)

    # deeplabtrain
    if event == 'DEEPLABTRAIN':
        start_deeplab()
        index = 2
    if event == 'STOPSTRAIN2':
        name = "main_ship.py"
        kill_train(name)
        index = 0
    # if event == 'DEEPLABTRAINALL':
    #     start_deeplab_train_all()
window.close()
