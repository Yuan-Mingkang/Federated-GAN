import os
path_root = os.getcwd()
Path = '/home/ilab/myself/project/Client/data/train/'
img_dir = os.listdir(Path)
for img in img_dir:
    if img.endswith('.png'):
        PngPath = Path + img
        img = img.replace("_labels", "")
        Newpath = Path + img
        os.rename(PngPath, Newpath)
    if img.endswith('.jpg'):
        PngPath = Path + img
        img = img.replace("_image", "")
        Newpath = Path + img
        os.rename(PngPath, Newpath)
