import time

import mxnet as mx
import numpy as np

from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
from getimgdata import GetImgData
import os,cv2

class CaptureFace(object):
    def __init__(self, detector, getdata, imgdir='./test_img', grayfacedir='./test_img_gray'):
        """
        初始化
        Args:
           detector: mtcnn 的人脸检测器
           getdata:获取图片数据类对象
           imgdir: 采集到的图片保存路径
           grayfacedir: 灰度处理后的图片的保存路径
        """
        self.detector = detector
        self.getdata = getdata
        self.imgdir = imgdir
        self.grayfacedir = grayfacedir

    def captureface(self, someone='someone', picturenum=10, waitKey=300):
        # 路径
        filepath = os.path.join(self.imgdir, someone)
        # 是否需要创建路径
        os.path.exists(filepath) or os.makedirs(filepath)
        # 调用电脑摄像头
        capture = cv2.VideoCapture(0)
        # 摄像头预热,避免出现曝光问题
        time.sleep(1)
        for i in range(picturenum):
            # 读取一帧
            ret, img = capture.read()
            # 翻转摄像头
            img = cv2.flip(img, 1)
            # 显示图像
            cv2.imshow('img', img)
            # 设置waitkey
            if cv2.waitKey(waitKey) == 27:  # 按esc退出
                break
            # 完整路径名
            fullpath = os.path.join(filepath, f'{i}.jpg')
            # 将图片写入指定路径
            cv2.imwrite(fullpath, img, [cv2.IMWRITE_JPEG_QUALITY, 100])  # IMWRITE_JPEG_QUALITY 代表jpeg图片的压缩质量
        capture.release()
        cv2.destroyAllWindows()

    def faceToGray(self, someone, size=64, waitKey=100):
        imgnames = self.getdata.getimgnames(os.path.join(self.imgdir,someone))
        n = len(imgnames)
        newpath = os.path.join(self.grayfacedir, someone)
        os.path.exists(newpath) or os.makedirs(newpath)
        for i in range(n):
            img = cv2.imread(imgnames[i])
            res = self.detector.detect_face(img)
            if res:
                faceboxes = res[0]
                index = np.sum(faceboxes < 0, axis=1) == 0
                faceboxes = faceboxes[index, :]
                # 遍历所有的人脸
                for x1, y1, x2, y2, score in faceboxes:
                    # 截取图片中的人脸图像
                    face = img[int(y1):int(y2) + 1, int(x1):int(x2) + 1]
                    # 转灰度图
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    # 压缩成64x64大小
                    face_gray = cv2.resize(face_gray, (size, size))
                    # 显示人脸图像
                    cv2.imshow('img', face_gray)
                    # 保存灰度处理之后的人脸
                    cv2.imwrite(f'{newpath}/{str(i)}.jpg', face_gray)
                    if cv2.waitKey(waitKey) == 27:
                        break
            cv2.destroyAllWindows()
            print('gray_finshed')


if __name__ == '__main__':
    detector = MtcnnDetector(model_folder='./mxnet_mtcnn_face_detection/model',
                             num_worker=4,
                             accurate_landmark=False,
                             ctx=mx.cpu())
    getdata = GetImgData()
    picture = CaptureFace(detector, getdata)
    picture.faceToGray('wo')
