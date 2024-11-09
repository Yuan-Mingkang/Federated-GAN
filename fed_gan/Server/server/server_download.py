import multiprocessing as mp
import gzip
from multi_thread_network.asyn_network import AsynServer
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int)
    parser.add_argument('--client_num',type=int)
    args = parser.parse_args()
    port = args.port
    client_num = args.client_num
    s = AsynServer(port, num_client=client_num, manager=mp.Manager())
    print("serverdownload")
    s.start()
    objects = dict()
    receive_objects = dict()
    while True:
        for i in range(client_num):
            objects[i] = None
            receive_objects[i] = None

        while len(receive_objects) != 0:
            s.asyn_receive_object(receive_objects)
            print(f"receive:{receive_objects}")
            for key in receive_objects:
                if receive_objects[key] is not None:
                    objects[key] = receive_objects[key]
                    f = open("../checkpoints/deeplab/best_deeplabv3plus_resnet101_ship_os16.pth", "rb")
                    file = f.read()
                    f.close()
                    s.asyn_send_object({key:file})

            for key in objects:
                if objects[key] is not None and key in receive_objects:
                    del receive_objects[key]

    s.close()

