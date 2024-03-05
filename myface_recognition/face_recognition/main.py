from mxnet_mtcnn_face_detection.mtcnn_detector import MtcnnDetector
from getimgdata import GetImgData
from sklearn.model_selection import train_test_split
from cnn_net2 import CnnNet
import mxnet as mx
import numpy as np
import cv2


def main(size=64, threshold=0.4, waitKey=1000):
    # 创建人脸检测器对象
    detector = MtcnnDetector(model_folder='./mxnet_mtcnn_face_detection/model',
                             num_worker=4,
                             accurate_landmark=False,
                             ctx=mx.cpu())
    # 读取图像信息
    getdata = GetImgData('./small_img_gray')
    x, y, number_name = getdata.readimg()
    output_size = len(number_name)

    # 数据分离
    # random_state：随机种子数，通过固定的random_state值，每次可以分到同样的训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

    # 模型测试
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, img = cap.read()
        # 人脸检测
        result = detector.detect_face(img)
        if result:
            faceboxes = result[0]
            index = np.sum(faceboxes < 0, axis=1) == 0  # array([False, False])
            # 将不全的人脸数据删除
            faceboxes = faceboxes[index, :]
            # 遍历所有的人脸
            for b in faceboxes:
                face = img[int(b[1]):int(b[3] + 1), int(b[0]):int(b[2] + 1)]
                # 转灰度图
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                # 压缩成指定大小
                face_gray = cv2.resize(face_gray, (size, size))
                face = face_gray.reshape((1,size,size,1))
                # 创建CnnNet卷积神经网络
                cnnnet = CnnNet(output_size=output_size)
                res, pre = cnnnet.cnn_predict(face)
                if np.max(res) < threshold:
                    name = 'unknown'
                else:
                    name = number_name[pre[0]]
                print('这个人是: %s' % name)
                # 在图片上显示名字
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]+1), int(b[3]+1)), (0, 0, 255))
                cv2.putText(img, name, (int(b[0]), int(b[1]-20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
                cv2.imshow('image', img)
                cv2.moveWindow('image',255,255)
            if cv2.waitKey(waitKey) == 27:  # Esc
                break
        # 释放摄像头
        cap.release()
        # 关闭显示窗口
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()