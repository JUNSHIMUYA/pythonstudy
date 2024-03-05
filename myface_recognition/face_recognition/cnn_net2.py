import os

import numpy as np
from sklearn.model_selection import train_test_split

from getimgdata import GetImgData
import tensorflow as tf

class CnnNet(object):
    def __init__(self, output_size, size=64, rate=0.5, filename='cnn_model.h5'):
        '''
        初始化 CnnNet 对象
        Args:
            output_size:输出层的节点数量（分类数量）
            size: size=64，代表图片的尺寸 64x64
            rate: dropout层的丢弃率，这里设置成0.5
            filename: 训练结果保存的文件
        '''
        self.output_size = output_size
        self.size = size
        self.rate = rate
        self.filename = filename

    def cnnLayer(self):
        '''构建模型'''
        # 实例化 Sequential 对象，使用add方法添加中间层
        model = tf.keras.Sequential()

        # 添加二维卷积层
        # -参数1：filters，输出空间的维度
        # -参数2：kernel_size，卷积核大小，是一个长度为2的元组
        # -参数3：strides，表示高度和宽度卷积的步长，可以用一个整数，或者用两个整数的元组
        # -参数4：padding，两个可选值，一个 valid（比较少用），另外一个 same
        # -参数5：activation，设置为relu函数
        # -参数6：输入维度 32 x 32 x 1
        model.add(tf.keras.layers.Conv2D(32, (3, 3),
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         input_shape=[self.size, self.size, 1],
                                         name='conv1',
                                         kernel_initializer='he_normal'))
        # 添加池化层，使用最大法池化
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same', name='pool1'))
        # 添加 dropout 层
        model.add(tf.keras.layers.Dropout(rate=self.rate, name='d1'))
        # 批量标准化正态分布
        model.add(tf.keras.layers.BatchNormalization())

        # +添加第二层卷积*
        model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         input_shape=[self.size, self.size, 1],
                                         name='conv2',
                                         kernel_initializer='he_normal'))
        # -添加第三层池化-
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same', name='pool2'))
        # +添加第二层的dropout*
        model.add(tf.keras.layers.Dropout(rate=self.rate, name='d2'))
        # +批量标准化正态分布*
        model.add(tf.keras.layers.BatchNormalization())

        # +添加第三层卷积-
        model.add(tf.keras.layers.Conv2D(64, (3, 3),
                                         strides=1,
                                         padding='same',
                                         activation='relu',
                                         input_shape=[self.size, self.size, 1],
                                         name='conv3',
                                         kernel_initializer='he_normal'))
        # +添加第三层的池化*
        model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=2, padding='same', name='pool3'))
        # +添加第三层的dropout-
        model.add(tf.keras.layers.Dropout(rate=self.rate, name='d3'))
        # +批量标准化正态分布*
        model.add(tf.keras.layers.BatchNormalization())

        # 全连接层
        model.add(tf.keras.layers.Flatten(name='flatten'))  # 展开
        model.add(tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'))
        #
        model.add(tf.keras.layers.Dropout(rate=self.rate, name='d4'))
        model.add(tf.keras.layers.Dense(self.output_size, activation='softmax', kernel_initializer='he_normal'))

        # 进行编译
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        return model

    def cnnTrain(self, x_train, y_train, retrain=True):
        '''
        根据训练样本训练模型
        Args:
            x_train:
            y_train:
            retrain: 便于调试
        Returns:
        '''
        if retrain:
            model = self.cnnLayer()
            batch_size = 100  # 每个人100张照片
            epochs = self.output_size * batch_size
            # 把训练集数据拆分成5折，1/5作为验证集，4/5作为训练集
            # verbose=2，每迭代一次输入一条日志
            # validation_split=0.2 拿训练数据的 1/5 作为验证集
            model.fit(x_train, y_train, batch_size=batch_size, verbose=2, epochs=epochs, validation_split=0.2)
            model.save(self.filename)
        else:
            if not os.path.exists(self.filename):
                print('文件%s不存在' % self.filename)

    def cnn_predict(self, x_test):
        '''
        Args:
            x_test: 测试集数据
        Returns:返回一个元组，包含概率数组和标签
        '''
        if not os.path.exists(self.filename):
            print('文件 %s 不存在' % self.filename)
        else:
            # 加载训练好的模型
            pre_model = tf.keras.models.load_model(self.filename)
            pro = pre_model.predict(x_test)
            print(pro)
            pre = np.argmax(pro, axis=1)
            return pro, pre



if __name__ == '__main__':
    # 路径
    path = './small_img_gray'
    getdata = GetImgData(dir=path)
    imgs, labels, number_names = getdata.readimg()
    # 标签个数
    output_size = len(number_names)
    # 创建神经网络
    cnnnet = CnnNet(output_size)
    # 数据分离
    # random_state=10，随机种子数，通过固定的值，每次可以分到同样的训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.2, random_state=10)
    cnnnet.cnnTrain(x_train, y_train)
    # 注意要创建神经网络对象，因为训练后和原神经网络会有误差
    cnnnet = CnnNet(output_size)
    pro, pre = cnnnet.cnn_predict(x_test)
    # 计算准确率
    acc_test = np.mean(np.argmax(y_test, axis=1) == pre)
    print(acc_test)