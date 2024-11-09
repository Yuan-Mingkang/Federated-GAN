# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys
import torch
from PIL import Image

fake_path = '/home/poac/4TB/yuanmingkang/city_result_data/fed_gan/epoch_test_data/fake_B_test_1/chicago23_labels.png'
# img = cv2.imread()
# img = cv2.imread('../data/dataset/fake/chicago354.png')
img = Image.open(fake_path)
img = np.array(img)
print(img)

# print(sys.getsizeof(img))

#取值范围：0~9，数值越小，压缩比越低，图片质量越高
# params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
params = [cv2.IMWRITE_JPEG_QUALITY, 70]
_, img_encode = cv2.imencode('.jpg', img, params)
print(sys.getsizeof(img_encode))
data_encode = np.array(img_encode)
str_encode = data_encode.tostring()

# 缓存数据保存到本地，以txt格式保存
with open('../data/dataset/img_encode.txt', 'wb') as f:
    f.write(str_encode)
    f.flush

with open('../data/dataset/img_encode.txt', 'rb') as f:
    str_encode = f.read()


nparr = np.frombuffer(str_encode, np.uint8)
img_decode = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
fake_img = torch.tensor(img_decode)
# fake_img = torch.tensor(img_decode).permute(2, 0, 1)
print(fake_img)
print(fake_img.shape)


