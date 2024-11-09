import os, signal

def start_gan_train(msg):
    port = 50012
    batchsize = msg['batch_size']
    learnrate = msg['learn_rate']
    load_size = msg['load_size']
    crop_size = msg['crop_size']
    epoch = int(msg['epoch'])
    client_num = msg['client_num']
    shell_msg = f"python ./server_train.py --model generator --lr {learnrate} --client_num {client_num} --epoch {epoch} --batch_size {batchsize} --load_size {load_size} --crop_size {crop_size} --port {port}  --preprocess resize_and_crop --dataset_mode city_train &"
    print(f"mask shell_msg {shell_msg}")
    os.system(shell_msg)

def kill_train(name):
    for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        print(pid)
        os.kill(int(pid), signal.SIGKILL)
    print("Process Successfully terminated")

if __name__ == '__main__':
    # kill_train('50012')
    # kill_train('50013')
    # kill_train('50014')
    msg = {}
    batchsize = 1
    learnrate = 0.0002
    imgsize = 512
    epoch = 2
    client_num = input("client_num:")
    msg['batch_size'] = batchsize
    msg['learn_rate'] = learnrate
    msg['load_size'] = 512
    msg['crop_size'] = imgsize
    msg['epoch'] = epoch
    msg['client_num'] = client_num

    start_gan_train(msg)