import os
import time
from PIL import Image

data_train_root = '/home/ilab/gxy/images/train_val/images'
data_gentator_root = '/home/ilab/gxy/images/train_2'
save_file_root = '/home/ilab/gxy/images/init_train_val/images'

file_names = os.listdir(data_train_root)
print(f"images have imgs str(len(file_names)")

for i,file_name in enumerate(file_names):
    print(f"process image {str(i)}")
    if file_name.endswith('.png'):
        imagename = file_name.split('.')[0]

        fake_read_name = str(imagename) + '.jpg'
        fake_save_name = str(imagename) + '.jpg'

        # read and save fake
        fake = Image.open(os.path.join(data_gentator_root,fake_read_name))
        fake.save(os.path.join(save_file_root,fake_save_name))

        # read and save mask
        mask = Image.open(os.path.join(data_train_root,file_name))
        mask.save(os.path.join(save_file_root,file_name))





# index = 0
# train_index = 0
# flag = 0
# file_len = len(file_names)
# print(f"dataset have images {str(file_len)}")
# path = os.path.join(save_file, 'train_' + str(train_index))
#
# t1  = time.time()
# for i,file_name in enumerate(file_names):
#
#     if file_name.endswith('.png'):
#         print(f"process index {str(index)}")
#         index = index + 1
#         if index > file_len/2*0.2 and flag == 0:
#             flag=1
#             train_index = train_index + 1
#             path = os.path.join(save_file, 'train_' + str(train_index))
#             data_root = '/home/ilab/gxy/AsynDGAN-master/results/example/train_5/images'
#
#
#
#         if flag == 0:
#             img_name = file_name.split('.')[0]
#
#             img_name = str(img_name) + '.jpg'
#
#             mask_path = os.path.join(path, file_name)
#             file_path = os.path.join(path, img_name)
#
#             file = Image.open(os.path.join(data_root, img_name))
#             file.save(file_path)
#
#             mask = Image.open(os.path.join(data_root, file_name))
#             mask.save(mask_path)
#         else:
#             img_name = file_name.split('.')[0]
#             img_read_name = img_name+"_fake_B.png"
#             img_name_1 = img_name
#             img_name = str(img_name) + '.jpg'
#
#             file_path = os.path.join(path, img_name)
#             mask_path = os.path.join(path, file_name)
#
#             file = Image.open(os.path.join(data_root, img_read_name))
#             file.save(file_path)
#
#             mask = Image.open(os.path.join('/home/ilab/gxy/images/train_2', file_name))
#             mask.save(mask_path)
# t2 = time.time()
# print(t2-t1)

#
# python images.py --dataroot /media/ilab/aa4924cc-5cd3-4a4f-995c-4721b4a821ac/sbk/data/ship --name ship_split_file --model asyndgan --netG resnet_9blocks --direction AtoB --lambda_L1 100 --dataset_mode ship_split --pool_size 0 --gpu_ids 0 --batch_size 4 --num_threads 1 --preprocess scale_width --load_size 512 --crop_size 512