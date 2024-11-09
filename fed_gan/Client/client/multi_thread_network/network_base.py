import socket
import os
import sys
import torch
import pickle
import time
import gzip


# 该方法用来计算 函数 运行时间
def time_compute(func):
    def wrapper(*args, **kwargs):
        s_time = time.time()
        ret = func(*args, **kwargs)
        e_time = time.time()
        print(f'execute function {func.__name__} cost {e_time - s_time:.3f} second!')
        return ret

    return wrapper


# 客户端类
class Client(object):
    def __init__(self, ip, port):
        self.ip = ip  # 服务端 ip
        self.port = port  # 服务端 port
        self.socket = socket.socket()  # 默认ipv4，tcp协议
        self.head_buffer_size = 1024  # 字符串一次接受最大长度
        self.var_buffer_size = 1048576  # var 变量 一次接受的最大长度

    def start(self):
        self.socket.connect((self.ip, self.port))  # 尝试与服务器建立socket套接字

    def send_head(self, length):  # 告诉server 传输信息的长度
        self.socket.send(f'{length:0>{self.head_buffer_size}}'.encode('utf-8'))

    def receive_head(self):  # 接受 server 提前传回的要发送的消息的长度
        length = b''
        length += self.socket.recv(self.head_buffer_size)
        while len(length) != self.head_buffer_size:
            length += self.socket.recv(self.head_buffer_size - len(length))
        return int(length)

    def send_msg(self, msg):  # 发送 str 类型消息
        self.send_head(len(msg))
        self.socket.send(msg.encode('utf-8'))

    @time_compute
    def send_object(self, variable):  # 发送 mask 给服务端
        variable = pickle.dumps(variable)

        # gzip

        variable = gzip.compress(variable)

        var_size = len(variable)
        message = f"(amount_data_send:{var_size})"
        with open('../loss/amount.txt', 'a') as log_file:
            log_file.write('%s\n' % message)
        self.send_head(var_size)

        send_size = 0
        while send_size < int(var_size):
            self.socket.send(variable[send_size:send_size + self.var_buffer_size])
            send_size += self.var_buffer_size

    @time_compute
    def receive_msg(self):  # 接受server 发来的 str 消息
        length = self.receive_head()
        msg = b''
        msg = self.socket.recv(length)
        while len(msg) < length:
            msg += self.socket.recv(length - len(msg))
        return msg

    def receive_object(self):  # 接受server 发来的 fake_image
        var_size = self.receive_head()
        message = f"(amount_data_recv:{var_size})"
        with open('../loss/amount.txt', 'a') as log_file:
            log_file.write('%s\n' % message)
        variable = b''
        while len(variable) != var_size:
            recv_size = min(self.var_buffer_size, var_size - len(variable))
            variable += self.socket.recv(recv_size)

        variable = gzip.decompress(variable)  # decompress

        variable = pickle.loads(variable)
        return variable

    def close(self):
        self.socket.close()


# 服务端类
class Server(object):
    def __init__(self, port, num_client=5):
        self.port = port  # 服务端的端口号
        self.socket = socket.socket()  # 创建监听socket（listenSocket）
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.socket.bind(('', self.port))  # 绑定服务端的端口号
        self.socket.listen()  # 将socket 设置成listenSocket ，准备监听客户端发来的创建dataSocket的连接
        self.clients = list()  # 客户端 dataSocket 列表
        self.num_client = num_client  # 客户端连接（dataSocket
        self.current_num_client = 0  # 目前存活的的客户端连接
        self.head_buffer_size = 1024
        self.var_buffer_size = 10485760

    def start(self):
        for i in range(self.num_client):  # 顺序创建 客户端数据连接
            client, addr = self.socket.accept()
            self.clients.append(client)
            self.current_num_client += 1

    def send_head(self, idx, length):  # 发送打招呼的信息，主要是通知对方接下来服务端发送信息的长度
        self.clients[idx].send(f'{length:0>{self.head_buffer_size}}'.encode('utf-8'))

    def receive_head(self, idx):  # 接受客户端的打招呼的信息
        length = b''
        length += self.clients[idx].recv(self.head_buffer_size)
        while len(length) != self.head_buffer_size:
            length += self.clients[idx].recv(self.head_buffer_size - len(length))
        return int(length)

    def send_msg(self, idx, msg):  # 服务端发送 str 类型消息
        self.send_head(idx, len(msg))
        self.clients[idx].send(msg.encode('utf-8'))

    @time_compute
    def send_object(self, idx, variable):  # 服务端发送 fake_image(主要是tensor类型的数据）
        variable = pickle.dumps(variable)

        # gzip.compress
        variable = gzip.compress(variable)

        var_size = len(variable)
        self.send_head(idx, var_size)

        send_size = 0
        while send_size < int(var_size):
            self.clients[idx].send(variable[send_size:send_size + self.var_buffer_size])
            send_size += self.var_buffer_size

    @time_compute
    def receive_object(self, idx):  # 服务端接受客户端发来的 mask
        var_size = self.receive_head(idx)

        variable = b''
        while len(variable) != var_size:
            recv_size = min(self.var_buffer_size, var_size - len(variable))
            variable += self.clients[idx].recv(recv_size)

        variable = gzip.decompress(variable)

        variable = pickle.loads(variable)
        return variable

    def receive_msg(self, idx):  # 服务端接受客户端发来的 str 消息
        msg_size = self.receive_head(idx)
        msg = self.clients[idx].recv(msg_size)
        while msg_size != len(msg):
            msg += self.clients[idx].recv(msg_size - len(msg))
        return msg

    def close(self):
        self.socket.close()


if __name__ == "__main__":
    # split the sum(x1 * x1) = z to client do x1 * x1 = temp_z, server do sum(temp_z) = z, transfer the grad
    if sys.argv[1] == 'client':
        print("Connecting test...")
        c = Client('127.0.0.1', 9092)
        c.start()
        print("pass!")

        print("message test...")
        c.send_msg('hello')
        print(c.receive_msg())
        print("pass!")

        print("variable test...")
        var = torch.randn((1000, 1000, 3), requires_grad=True)
        c.send_object(var)
        var = c.receive_object()
        # print(var)
        print("pass!")

        print('training test...')
        for i in range(100):
            temp_z = var * var
            c.send_object(temp_z)
            grad = c.receive_object()
            print('grad is', grad)
            f = grad * temp_z
            f = f.sum()
            f.backward()
            print('var grad', var.grad)
            var = var - 0.1 * var.grad
            var = var.detach()
            var.requires_grad = True
            # var.grad.data.zero_()

        c.close()

    if sys.argv[1] == 'server':
        s = Server(9092, num_client=1)
        print("Connecting test & message test...")
        s.start()
        print(s.receive_msg(0))
        s.send_msg(0, 'hello client')
        print("pass!")

        print("variable test...")
        var = s.receive_object(0)
        # print(var)
        s.send_object(0, var)
        print("pass!")

        print('training test...')
        for i in range(100):
            temp_z = s.receive_object(0)
            z = temp_z.sum()
            z.backward()
            s.send_object(0, temp_z.grad)
            print(z)

        s.close()
