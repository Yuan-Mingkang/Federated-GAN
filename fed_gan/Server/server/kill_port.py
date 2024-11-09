import os,signal

def killport(port):
    for line in os.popen("ps ax | grep " + str(port) + " | grep -v grep"):
        fields = line.split()
        pid = fields[0]
        print(pid)
        os.kill(int(pid), signal.SIGKILL)
        print("kill successful")

if __name__ == '__main__':
    killport(50012)
    killport(50013)
    killport(50014)