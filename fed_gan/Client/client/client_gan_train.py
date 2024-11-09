import sys
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from multi_thread_network.network_base import Client
from util.zip_unzip import *
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
import PySimpleGUI as sg
# from torch.utils.tensorboard import SummaryWriter

#  receive paremeters from server
# def receive_paremeters(client,opt):
#     msg = client.receive_object()
#     opt.batch_size = msg['batch_size']
#     opt.lr = msg['learn_rate']
#     opt.load_size = msg['load_size']
#     opt.crop_size = msg['crop_size']
#     opt.epoch = int(msg['epoch'])
#     return opt


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


if __name__ == '__main__':

    opt = TrainOptions().parse()
    server_ip = opt.server_ip
    port = opt.port
    print(f"client_gan_train mask_path:{opt.mask_path}")
    print(type(opt.mask_path))
    if opt.mask_path == 'None':
        sg.popup_auto_close("Please enter mask_path and start again!", title='tishi', button_type=5, font='Any 18')
    try:
        client = Client(server_ip, port)
        client.start()
    except Exception as e:
        sg.popup_auto_close("Please click the [StartTrain] button on the server side first and start again!", title='tishi', button_type=5, font='Any 18')

    msg = client.receive_object()
    opt.batch_size = msg['batch_size']
    # opt.lr = msg['learn_rate']
    opt.load_size = msg['load_size']
    opt.crop_size = msg['crop_size']
    opt.epoch = int(msg['epoch'])
    opt.D_learningrate = msg['D_learningrate']
    opt.Update_Frequency = msg['Update_Frequency']
    print(opt.Update_Frequency)
    print(type(opt.Update_Frequency))
    print(f"msg{msg}")


    # loading dataset
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    dataset_class = find_dataset_using_name(opt.dataset_mode)
    dataset_real = dataset_class(opt)


    if opt.Update_Frequency > 0:

        opt.model = 'discriminator_epoch'
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        torch.backends.cudnn.benchmark = False
        # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
        print("pass")
        total_iters = 0  # the total number of training iterations

        # transport mask
        # 87-99line compress all masks to zip_mask,and delivery zip_maks to server
        raw_mask_path = opt.mask_path
        print(raw_mask_path)
        zip_mask_path = "/".join(raw_mask_path.split('/')[:-1])

        zip_mask_path = os.path.join(zip_mask_path, 'mask_zip.zip')
        # mask compress zip
        zip_dir(raw_mask_path, zip_mask_path)
        # read mask_zip file
        f = open(zip_mask_path, 'rb')
        zip_mask = f.read()
        f.close()
        client.send_object(zip_mask)  #
        print("transport zip_mask over ")
        sg.popup_auto_close("Sending data finished, start training!", title='tishi', button_type=5, font='Any 18')
        update_learn_rate = client.receive_object()
        print(f"update_learn_rate{str(update_learn_rate)}")
        # model.load_networks("latest")
        training_start = time.time()
        # model.load_networks("700")
        for epoch in range(
                opt.epoch):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            while True:
                training = client.receive_object()
                print(training['training'])
                if training['training']:
                    break
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            batchsize = int(opt.batch_size)
            length = dataset_real.__len__()
            a = length // batchsize
            epoch_loss_D_report = 0
            epoch_loss_D_fake_report = 0
            epoch_loss_D_real_report = 0
            epoch_loss_G_report = 0
            epoch_loss_G_GAN_report = 0
            epoch_loss_G_perceptual_report = 0
            for i in range(length // batchsize):  # inner loop within one epoch
                sys.stdout.flush()
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                # visualizer.reset()
                total_iters += 1
                epoch_iter += opt.batch_size
                loss_D_report, loss_D_fake_report, loss_D_real_report, loss_G_report, loss_G_GAN_report, loss_G_perceptual_report = model.optimize_parameters(
                    client, dataset_real, batchsize)  # calculate loss functions, get gradients, update network weights
                epoch_loss_D_report += loss_D_report
                epoch_loss_D_fake_report += loss_D_fake_report
                epoch_loss_D_real_report += loss_D_real_report
                epoch_loss_G_report += loss_G_report
                epoch_loss_G_GAN_report += loss_G_GAN_report
                epoch_loss_G_perceptual_report += loss_G_perceptual_report

                if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size

                iter_data_time = time.time()

            epoch_loss_D_report = epoch_loss_D_report / a
            epoch_loss_D_fake_report = epoch_loss_D_fake_report / a
            epoch_loss_D_real_report = epoch_loss_D_real_report / a
            epoch_loss_G_report = epoch_loss_G_report / a
            epoch_loss_G_GAN_report = epoch_loss_G_GAN_report / a
            epoch_loss_G_perceptual_report = epoch_loss_G_perceptual_report / a

            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            #     send epoch latest.epoch
            # model.update_learning_rate
            if total_iters % int(update_learn_rate) == 0:
                model.update_learning_rate()

            print('End of epoch %d / %d \t Time Taken: %d sec' % (
            opt.epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        training_stop = time.time()
        training_time = training_stop - training_start
        print('training_time : %.5f sec' % training_time)

        sg.popup_auto_close("Training completed!", title='tishi', button_type=5, font='Any 18')

    else:
        opt.model = 'discriminator'
        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        torch.backends.cudnn.benchmark = False
        # visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
        total_iters = 0  # the total number of training iterations

        # transport mask
        # 87-99line compress all masks to zip_mask,and delivery zip_maks to server
        raw_mask_path = opt.mask_path
        zip_mask_path = "/".join(raw_mask_path.split('/')[:-1])

        zip_mask_path = os.path.join(zip_mask_path, 'mask_zip.zip')
        # mask compress zip
        zip_dir(raw_mask_path, zip_mask_path)
        # read mask_zip file
        f = open(zip_mask_path, 'rb')
        zip_mask = f.read()
        f.close()
        client.send_object(zip_mask)  #
        sg.popup_auto_close("Sending data finished, start training!", title='tishi', button_type=5, font='Any 18')
        update_learn_rate = client.receive_object()
        print(f"update_learn_rate{str(update_learn_rate)}")
        for epoch in range(
                opt.epoch):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            batchsize = int(opt.batch_size)
            length = dataset_real.__len__()

            for i in range(length // batchsize):  # inner loop within one epoch
                sys.stdout.flush()
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
              # visualizer.reset()
                total_iters += 1
                epoch_iter += opt.batch_size
                model.optimize_parameters(client,
                                          dataset_real, batchsize)  # calculate loss functions, get gradients, update network weights

                losses = model.get_current_losses()

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                iter_data_time = time.time()
            print(f"total iter{str(total_iters)}")
            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)
            #     send epoch latest.epoch
            # model.update_learning_rate
            if total_iters % int(update_learn_rate) == 0:
                model.update_learning_rate()

            print('End of epoch %d / %d \t Time Taken: %d sec' % (
            opt.epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            # update learning rates at the end of every epoch.
        sg.popup_auto_close("Training completed!", title='tishi', button_type=5, font='Any 18')
        client.close()
        print("training completed")

