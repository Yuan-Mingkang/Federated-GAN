# # coding:utf-8

from options.test_options import TestOptions
from models import create_model
from multi_thread_network.asyn_network import *
import multiprocessing as mp
import torchvision.transforms as transforms

if __name__ == '__main__':
    opt = TestOptions().parse()
    port = opt.port
    client_num = opt.client_num
    print("start generate fake_image")
    opt.dataroot = None  # dataroot
    opt.num_threads = 1  # val code only supports num_threads = 1
    opt.batch_size = 1  # val code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen trainB are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped trainB are needed.
    opt.display_id = -1  # no visdom display; the val code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    s = AsynServer(port, num_client=client_num, manager=mp.Manager())
    s.start()

    objects = {key: None for key in range(client_num)}
    receive_objects = {key: None for key in range(client_num)}
    while len(receive_objects) != 0:
        s.asyn_receive_object(receive_objects)  # receive

        for key in receive_objects:
            if receive_objects[key] is None:
                continue

            mask_tensor = receive_objects[key]

            img = {}
            img['A'] = mask_tensor
            img['A_paths'] = ""
            model.eval()
            model.set_input(img)  # unpack data from data loader
            model.test()  # run inference
            visual = model.get_current_visuals()

            fake = visual['fake'][0]
            fake = (fake + 1) / 2
            unloader = transforms.ToPILImage()
            fake_img = unloader(fake)
            fake_img.show()
            s.asyn_send_object({key: fake_img})

            for key in list(receive_objects):
                if receive_objects[key] is not None:
                    objects[key] = receive_objects[key]
                    receive_objects[key] = None

    s.close()