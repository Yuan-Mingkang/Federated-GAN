import os,signal
import PIL.Image
import io
import base64
import PySimpleGUI as sg
import os.path
from deeplab_test import predict
import client_gan_test
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from multi_thread_network.network_base import Client
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# add draw figures 2022.08.05
def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def start_download(serverip):
    port = 50014
    shell_msg = f"python ./client_download.py --server_ip {serverip} --port {port} &"
    os.system(shell_msg)


# def start_deeplab_val(val_image_path):
#     for i in range(2):
#         if i == 0:
#             save_val = 'all'
#             ckpt = "../checkpoints/deeplab_city_all/best_deeplabv3plus_resnet101_city_os16.pth"
#             if not os.path.exists(ckpt):
#                 continue
#         else:
#             save_val = "asyndgan"
#
#             ckpt = "../checkpoints/deeplab_city/best_deeplabv3plus_resnet101_city_os16.pth"
#
#         val_image_path = val_image_path
#         shell_msg = f'python ../deeplab/eval_city.py --num_classes 3 --val_batch_size 1 --crop_size 512 --model deeplabv3plus_resnet101 --dataset city --file_dir {val_image_path} --ckpt {ckpt} --save_val {save_val}'
#         os.system(shell_msg)



def start_gan_train(serverip,mask_path):
    port = 50012
    shell_msg = f"python ./client_gan_train.py --server_ip {serverip} --port {port} --mask_path {mask_path} &"
    os.system(shell_msg)

def kill_train(name):
    for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)
    print("Process Successfully terminated")

def convert_to_bytes(file_or_bytes, resize=None):
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()

"""css"""
# 文本大小
TEXT_SIZE = (10, 1)
# 按钮大小
BUTTON_SIZE = (22, 1)
# 输入文本大小
INPUT_TEXT_SIZE = (12, 20)
IN_SIZE = (15,1)
# 窗口大小
WINDOWS_SIZE = (1366, 900)
# 文字样式及大小
FONT_H1_SIZE = ('宋体',12)
FONT_BUTTON_SIZE =  ('宋体',10)
FONT_TEXT_SIZE = 'Any 12'
SEARCH_BUTTON_SIZE = (12, 1)
LIST_BOX_SIZE = (42, 40)
MULTLINE_SIZE = (22, 25)
# 展示面版大小
IMAGE1_SIZE = (800,400)
# 展示loss界面大小
FIGURE_SIZE = (800,400)

# 展示图片大小
PIC_SIZE = [400,300]
"""css"""

# add loss display 2022.08.05
draw_pic = [
    # [sg.Text('loss', font='Any 18')],
    [sg.Canvas(key='canvas',size=FIGURE_SIZE,background_color='white')]
]
start_client =[
    [sg.Text('serverip',font=FONT_TEXT_SIZE,  size=TEXT_SIZE),sg.InputText(size=INPUT_TEXT_SIZE, key="SERVERIP", default_text='10.2.28.32')],
    [sg.Button('Connect', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE,key=("CONNECTSERVER"))],
]
gan = [
    [
        sg.In(size=IN_SIZE, enable_events=True, default_text='',key="-MASK-",),
        sg.FolderBrowse('searchFolder',size=SEARCH_BUTTON_SIZE),
    ],
    [sg.Button('StartTrain', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=("STARTTRAIN"))],
    [sg.Button('StopTrain', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE,  key=("STOPTRAIN"))],
    [sg.FileBrowse('SearchPicture', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE, key=('SEARCHPICTURE1'))],
    [sg.Button('SendPicture', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE,  key=("SENDPICTURE1"))],
]
gan_img = [
    [sg.Image(key="-IMAGE1-",size=IMAGE1_SIZE,background_color='white',expand_x=True,expand_y=True)],
]
file_list_column = [
    [sg.Button('ModelDownload', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE,  key=("MODELDOWNLOAD"))],
    [
        sg.In(size=IN_SIZE, enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse('SearchFolder',size=SEARCH_BUTTON_SIZE),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=LIST_BOX_SIZE, key="-FILE LIST-"
        )
    ],
]
# image_viewer_column = [
#     # [sg.Text(size=(500, 1), key="-TOUT-")],
#     [sg.Image(key="-IMAGE2-",background_color='white')],
# ]

deeplab_val = [
    [
        sg.In(size=IN_SIZE, enable_events=True, default_text='img_path', key="VAL_IMAGE", ),
        sg.FolderBrowse('浏览',size=SEARCH_BUTTON_SIZE),
    ],
        [sg.Button('val', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE,  key=("VAL"))],
        [sg.Button('show_result', size=BUTTON_SIZE,font=FONT_BUTTON_SIZE,  key=("SHOW_RESULT"))],
        [sg.Multiline('', key='_Multiline_', size=MULTLINE_SIZE)]
]
layout = [
    [sg.Frame('ClientInit',start_client, title_color='black', font=FONT_H1_SIZE)],
    [sg.Frame('Gan', gan, title_color='black', font=FONT_H1_SIZE) ,sg.Frame('Loss', draw_pic, title_color='black',background_color='white',pad=((200,0),(0,0)),size=FIGURE_SIZE, font=FONT_H1_SIZE)],
    [sg.Frame('Deeplab', file_list_column, title_color='black', font=FONT_H1_SIZE),
    # sg.Frame('SegmentationResults', image_viewer_column, title_color='black', font=FONT_H1_SIZE,pad=((200,0),(0,0))),
    # sg.Frame('DeeplabVal', deeplab_val, title_color='black', font=FONT_H1_SIZE)
sg.Frame("ShowResult",gan_img,title_color='black', font=FONT_H1_SIZE,pad=((200,0),(0,0)),size=IMAGE1_SIZE)
     ],

]
window = sg.Window('Client', layout, size=WINDOWS_SIZE, resizable=True,location=(0,0)).finalize()
plt.figure(1)
fig = plt.gcf()
DPI = fig.get_dpi()
fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)

first_delete_gan = 1
serverip = None
predictop = None
mask_path = None
client = None
gantest = None
val_image_path =None
first_delete_d_loss = True

# 2022.8.23添加展示loss功能
# loss_G,{loss_G_report},loss_G_GAN,{loss_G_GAN_report},loss_G_perceptual,{loss_G_perceptual_report}
LOSS_G_PATH = '../loss/loss_G.txt'
# loss_D,{loss_D_report},loss_D_fake,{loss_D_fake_report},loss_D_real,{loss_D_real_report},{training_D}
LOSS_D_PATH = '../loss/loss_D.txt'
while True:

    # 横坐标为存数据的频率，纵坐标为误差
    # 清除以往的训练数据，防止对画图进行干扰
    if first_delete_gan == 1:
        if os.path.exists(LOSS_D_PATH):
            os.remove(LOSS_D_PATH)
            os.remove(LOSS_G_PATH)
        os.system(f'touch {LOSS_D_PATH}')
        os.system(f'touch {LOSS_G_PATH}')
        first_delete_gan = 0
    # 读取D_loss
    with open(LOSS_D_PATH, "r") as log_file:
        content = log_file.readlines()
        iters = []
        loss_D = []
        loss_D_fake = []
        loss_D_real = []
        for i,line in enumerate(content):
            line = line[:]
            list_l = line.split(',')
            iters.append(i)
            loss_D.append(round(float(list_l[1]),2))
            loss_D_fake.append(round(float(list_l[3]),2))
            loss_D_real.append(round(float(list_l[5]),2))
    log_file.close()
    with open(LOSS_G_PATH, "r") as log_file:
        content = log_file.readlines()
        iters_G = []
        loss_G = []
        loss_G_GAN = []
        loss_G_perceptual = []
        for i, line in enumerate(content):
            line = line[:]
            list_l = line.split(',')
            iters_G.append(int(i))
            loss_G.append(round(float(list_l[1]),2))
            loss_G_GAN.append(round(float(list_l[3]),2))
            loss_G_perceptual.append(round(float(list_l[5]),2))
    log_file.close()
        # plt.cla()
    loss_D = [i*10 for i in loss_D]
    loss_D_plt, = plt.plot(iters,loss_D,color='red')
    # loss_D_real_plt, = plt.plot(iters,loss_D_real,color='green')
    # loss_D_fake_plt, = plt.plot(iters,loss_D_fake,color='blue')
    loss_G_plt, = plt.plot(iters_G,loss_G,color='black')
    # loss_G_GAN_plt, = plt.plot(iters_G,loss_G_GAN,color='yellow')
    # loss_G_perceptual_plt, = plt.plot(iters_G,loss_G_perceptual,color='gray')

    # plt.legend([loss_D_plt,loss_D_real_plt,loss_D_fake_plt,loss_G_plt,loss_G_GAN_plt,loss_G_perceptual_plt],
    #            ['loss_D','loss_D_real','loss_D_fake','loss_G','loss_G_GAN','loss_G_perceptual'])
    plt.legend([loss_D_plt,loss_G_plt],
               ['loss_D','loss_G'])

    fig_canvas_agg.draw()

    event, values = window.read(timeout=500)
    if event == None:
        break
    if event == 'VAL_IMAGE':
        val_image_path =  values['VAL_IMAGE']
        print(val_image_path)
    # 增加了服务端是否开启判断
    if event == 'CONNECTSERVER':
        serverip = values['SERVERIP']
        port_fake_img = 50013
        client = Client(serverip,port_fake_img)
        error = None
        try:
            client.start()
        except Exception as e:
            error = e
            sg.popup_auto_close('Server not turned on!', title='提示', button_type=5, font='Any 18')
        else:
            sg.popup_auto_close('Connection Successfully!', title='提示', button_type=5, font='Any 18')
            gantest = client_gan_test.GanTest(client)

    if event == 'STARTTRAIN':

        start_gan_train(serverip,mask_path)

    if event =='STOPTRAIN':
        name = "client_gan_train.py"
        kill_train(name)
    if event == '-MASK-':
        mask_path = values['-MASK-']
        print(f"mask_path :{mask_path}")
    # gan test
    if event =='SENDPICTURE1':
        mask_path = values['SEARCHPICTURE1']
        savepath = '/'.join(mask_path.split('/')[:-1])
        savepath = os.path.join(savepath,'fakeimg')
        filename = str(mask_path.split('/')[-1])
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        save_path = os.path.join(savepath,filename)

        gantest.start(mask_path, save_path )
        window["-IMAGE1-"].update(convert_to_bytes(save_path, PIC_SIZE))

    # download model
    if event == 'MODELDOWNLOAD':
        start_download(serverip)

    # 2022.0806 deeplab文件夹直接复制以前的
    if event == "-FOLDER-":
        predictop = predict.Predict()

        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
               and f.lower().endswith((".jpg"))
        ]
        window["-FILE LIST-"].update(fnames)
    # 将分割的结果展示在测试集同一个父目录下predict文件中
    elif event == "-FILE LIST-":
        try:
            file_path = values["-FOLDER-"]
            file_name = values["-FILE LIST-"][0]
            save_path = file_path

            save_path = save_path.split('/')[:-1]
            save_path = "/".join(save_path)

            save_path = os.path.join(save_path,'predict')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            save_path = os.path.join(save_path,file_name)
            predictop.start(save_path,file_path,file_name)
            window["-IMAGE1-"].update(convert_to_bytes(save_path,PIC_SIZE))
        except:
            pass
window.close()

