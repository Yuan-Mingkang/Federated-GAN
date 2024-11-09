import os
import zipfile
import glob

def zip_dir(dir_path, zip_path):
    '''
    conpress mask

    :param dir_path: 目标文件夹路径
    :param zip_path: 压缩后的文件夹路径
    :return:
    '''
    i=1
    zip = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    for root, dirnames, filenames in os.walk(dir_path):
        file_path = root.replace(dir_path, '')  # 去掉根路径，只对目标文件夹下的文件及文件夹进行压缩
        # 循环出一个个文件名
        for filename in filenames:
            zip.write(os.path.join(root, filename), os.path.join(file_path, filename))
            #print(i)
            i+=1
    zip.close()

def unzip_file(dir_path,unzip_file_path):
    # 解压缩后文件的存放路径
    #unzip_file_path = r"C:\Users\Desktop\新建文件夹"
    # 找到压缩文件夹
    dir_list = glob.glob(dir_path)
    if dir_list:
        # 循环zip文件夹
        for dir_zip in dir_list:
            # 以读的方式打开
            with zipfile.ZipFile(dir_zip, 'r') as f:
                for file in f.namelist():
                    f.extract(file, path=unzip_file_path)
            # os.remove(dir_zip)

if __name__ == '__main__':
    unzip_file_path = r"C:\Users\Desktop\新建文件夹"
    zip_file_path = r"C:\Users\Desktop\新建文件夹\*.zip"
    unzip_file(zip_file_path, unzip_file_path)
    # 这儿的 dir_path 只是其中的一种路径处理方式，可以根据自己的需求行进实际操作