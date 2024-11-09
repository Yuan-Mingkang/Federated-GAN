from multi_thread_network.network_base import Client
from PIL import Image
import torchvision.transforms as transforms


class GanTest():
    def __init__(self,client):

        self.client = client

    def start(self,mask_path, save_path ):
        print("in client_test_start")
        mask = Image.open(mask_path).convert('RGB')
        loader = transforms.Compose([transforms.ToTensor()])
        mask_tensor = loader(mask).unsqueeze(0)   # 转变为Tensor格式

        self.client.send_object(mask_tensor)

        fake_img = self.client.receive_object()

        # transform = transforms.ToPILImage()
        # fakeimg = transform(fake_img)
        fake_img.save(save_path)
        print("fake save")
        # self.client.close()
