from multi_thread_network.network_base import Server, Client
import multiprocessing as mp
import sys
import torch

class AsynServer(Server):
    def __init__(self, port, num_client, manager):
        super().__init__(port, num_client=num_client)
        self.recieved_object = manager.dict()  # id --> list of object ;接受消息的 队列
        for i in range(num_client):
            self.recieved_object[i] = manager.list()#每个客户端dataSocket连接都有一个消息list
        self.receive_lock = manager.Lock()
        self.clients = manager.list(self.clients)#

    def start(self):
        super().start()#顺序创建dataSocket 连接
        self._init_process()
        return self.current_num_client

    def _init_process(self):#启用多进程，准备接受消息
        
        process = list()
        for i in range(self.num_client):
            p = mp.Process(target=self.receive_object_process, args=(i,))
            process.append(p)
        
        # start all the prcess
        [p.start() for p in process]
        return 

    
    def asyn_send_object(self, msg):#异步发送消息
        """
        Args: 
            msg: dict(), id --> Object
        """

        process = list()
        for idx, variable in msg.items():
			# 用于创建进程模块，process模块用来创建子进程，可以实现多进程的创建，启动和关闭等操作。
            p = mp.Process(target=self.send_object, args=(idx, variable,))
            process.append(p)
        
        # start up all the process，start()函数表示进程准备就绪，等待CPU调度。
        [p.start() for p in process]
        return 
    
    def asyn_receive_object(self, msg):#异步从消息队列里面取mask
        """
        Args:
            msg: dict(), id --> None
        
        Returns:
            mgs: dict(), id --> Object
        """
        
        for idx, variable in msg.items():
            assert (variable is None), f"{idx}:{variable} will be override!"
        
        # check if received_objects have any object
        

        for idx, object_list in self.recieved_object.items():
            if idx not in msg:
                continue

            if len(object_list) == 0:
                continue

            msg[idx] = object_list[0]
            self.receive_lock.acquire()
            del object_list[0]
            self.receive_lock.release()

        return
    
    def receive_object_process(self, idx):#异步接受客户端发来的mask,并存入消息队列里面
        """ Receive all the object from idx Client
        Args: 
            idx: id of clients
        """

        while True:
			# 创建一个字典对象，该字典对象是一个进程安全的共享字典。多个进程可以通过调用该字典对象的方法来对字典进行读写操作，保证进程间数据的同步和安全性。
            var = self.receive_object(idx)
			# 加锁，Lock是一个互斥锁，用于保护共享资源的访问。当一个进程获得了锁时，其他进程就无法获得锁，只能等该进程释放锁后才能继续执行，避免多个进程同时修改同一个对象。
            self.receive_lock.acquire()
			# append用于向共享列表添加一个元素。由于多个进程会同时调用该函数，因此需要使用Lock来保护共享列表的访问，避免出现并发修改问题。
            self.recieved_object[idx].append(var)
			# 解锁
            self.receive_lock.release()
        

if __name__ == '__main__':
    port = 60100
    client_num = 2
    # split the sum(x1 * x1) = z to client do x1 * x1 = temp_z, server do sum(temp_z) = z, transfer the grad


    s = AsynServer(port, num_client=client_num, manager=mp.Manager())
    print("Connecting val")
    s.start()#顺序创建与客户端通信的dataSocket
    print("pass!")

    print("variable val...")
    objects = dict()
    receive_objects = dict()
    for i in range(client_num):
        objects[i] = None
        receive_objects[i] = None

    while len(receive_objects) != 0:
        s.asyn_receive_object(receive_objects)
        for key in receive_objects:
            if receive_objects[key] is not None:
                objects[key] = receive_objects[key]
        for key in objects:
            if objects[key] is not None and key in receive_objects:
                del receive_objects[key]
        
    for key in objects:
        print(key, objects[key])

    s.asyn_send_object(objects)
    print("pass!")

    print('training val...')
    for i in range(100):
        print(i)
        objects = {key: None for key in range(client_num)}
        receive_objects = {key: None for key in range(client_num)}

        while len(receive_objects) != 0:
            s.asyn_receive_object(receive_objects)
            for key in receive_objects:
                if receive_objects[key] is None:
                    continue

                temp_z = receive_objects[key]
                z = temp_z.sum()
                z.backward()#相当于训练的过程
                s.asyn_send_object({key: temp_z.grad})
                print(f'{z} from {key}')

            for key in receive_objects:
                if receive_objects[key] is not None:
                    objects[key] = receive_objects[key]
            for key in objects:
                if objects[key] is not None and key in receive_objects:
                    del receive_objects[key]

    s.close()
