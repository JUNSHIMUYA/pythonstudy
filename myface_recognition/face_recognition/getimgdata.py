import numpy as np
import os
import cv2


class GetImgData(object):
    """ 获取图像文件的类 """

    def __init__(self, dir='./small_img_gray'):
        self.dir = dir

    def onehot(self, num_list):
        b = np.zeros((len(num_list), max(num_list) + 1))
        b[np.arange(len(num_list)), num_list] = 1
        print(b)
        return b.tolist()

    def getimgnames(self, path):
        imgNames = []
        filenames = os.listdir(path)
        for file in filenames:
            if file.endswith('.jpg') or file.endswith('.png'):
                fullPath = os.path.join(path, file)
                imgNames.append(fullPath)
        return imgNames

    def getfileandlabels(self):
        dir = self.dir
        dictdir = {}
        for name in os.listdir(dir):
            if os.path.isdir(os.path.join(dir, name)):
                dictdir[name] = os.path.join(dir, name)
        namelist = dictdir.keys()
        pathlist = dictdir.values()
        indexlist = list(range(len(namelist)))
        ret1 = dict(zip(pathlist, self.onehot(indexlist)))
        ret2 = dict(zip(indexlist, namelist))
        return ret1, ret2

    def readimg(self):
        """
        获取指定文件夹下的所有图片，取每张图片单个通道的值
        Returns 1:相当于训练时输入的x值，这里是图片的矩阵
        Returns 2:相当于训练时输入的y值，这里是图片的标签对应的编码
        Returns 3:标签与人名对应的字典
        """
        imgs = []  # 图片的像素数据
        labels = []  # 存放图片标签的列表
        # 获取dir中每个文件的独热编码，以及每个文件的标签
        dir_labels, number_names = self.getfileandlabels()
        for dirname, label in dir_labels.items():
            # 遍历文件名下所有的图片
            for imgname in self.getimgnames(dirname):
                ret = cv2.imread(imgname)
                # print(ret.shape)
                img = ret[:, :, 0:1]
                imgs.append(img)
                labels.append(label)
        # 将图片的数据归一化
        x = np.array(imgs, dtype='float32') / 255  # 值的范围确定为0~1
        y = np.array(labels, dtype='float32')
        return x, y, number_names


if __name__ == '__main__':
    getData = GetImgData()
    # getData.onehot([0, 1, 2, 5])
    ret = getData.getfileandlabels()
    x, y, number_names = getData.readimg()
    print('x shape:', x.shape)
    print('y shape', y.shape)
    print(number_names)
    # print(ret)
    # print(len(ret))
