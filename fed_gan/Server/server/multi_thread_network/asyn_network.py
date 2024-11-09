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
            p = mp.Process(target=self.send_object, args=(idx, variable,))
            process.append(p)
        
        # start up all the process
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
            var = self.receive_object(idx)
            self.receive_lock.acquire()
            self.recieved_object[idx].append(var)
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