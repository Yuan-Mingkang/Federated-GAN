import os, signal

def kill_train(name):
    for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        os.kill(int(pid), signal.SIGKILL)
    print("Process Successfully terminated")

if __name__ == '__main__':
    name = "client_gan_train_v2.py"
    kill_train(name)