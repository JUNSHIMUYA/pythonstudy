
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
#
# def change_alpha(img, a):
#     im_changed = np.zeros(shape=img.shape, dtype='uint8')
#     for i in range(img.shape[0]):
#         for j in range(img.shape[1]):
#             for k in range(img.shape[2]):
#                 if img[i, j, k] * a > 255:
#                     im_changed[i, j, k] = 255
#                 elif img[i, j, k] * a < 0:
#                     im_changed[i, j, k] = 0
#                 else:
#                     im_changed[i, j, k] = img[i, j, k] * a
#     return im_changed
#
#
# if __name__ == '__main__':
#     fyb = plt.imread('D:\\pycharmproject\\20180663张清辉_python\\course\\1.jpg')
#     high_p = change_alpha(fyb, 1.5)
#     low_p = change_alpha(fyb, 0.5)
#     mhigh_p = change_alpha(fyb, 3)
#     plt.figure(figsize=(5, 5))  # 设置窗口大小
#     plt.suptitle('实验1-1')  # 图片名称
#
#     plt.subplot(2, 2, 1), plt.title('a = 0.5')
#     plt.imshow(low_p), plt.axis('off')
#
#     plt.subplot(2, 2, 2), plt.title('原图')
#     plt.imshow(fyb), plt.axis('off')
#
#     plt.subplot(2, 2, 3), plt.title('a = 1.5')
#     plt.imshow(high_p), plt.axis('off')
#
#     plt.subplot(2, 2, 4), plt.title('a = 2')
#     plt.imshow(high_p), plt.axis('off')
#     plt.show()


# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
#
# # 支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
#
# def matrix_conv(arr, kernel):
#     n = len(kernel)
#     ans = 0
#     for i in range(n):
#         for j in range(n):
#             ans += arr[i, j] * float(kernel[i, j])
#     return ans
#
#
# def conv2d(img, kernel):
#     n = len(kernel)
#     img1 = np.zeros((img.shape[0] + 2 * (n - 1), img.shape[1] + 2 * (n - 1)))
#     img1[(n - 1):(n + img.shape[0] - 1), (n - 1):(n + img.shape[1] - 1)] = img
#     img2 = np.zeros((img1.shape[0] - n + 1, img1.shape[1] - n + 1))
#     for i in range(img1.shape[0] - n + 1):
#         for j in range(img1.shape[1] - n + 1):
#             temp = img1[i:i + n, j:j + n]
#             img2[i, j] = matrix_conv(temp, kernel)
#     new_img = img2[(n - 1):(n + img.shape[0] - 1), (n - 1):(n + img.shape[1] - 1)]
#     return new_img
#
#
# kernel5 = np.ones((9, 9)) / (9 ** 2)
# img = plt.imread('D:\\pycharmproject\\20180663张清辉_python\\course\\1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img5 = conv2d(img, kernel5)
#
# plt.figure("1-4", figsize=(8, 8))
# plt.subplot(121)
# plt.title('原始图像')
# plt.imshow(img, cmap=plt.cm.gray)
#
# plt.subplot(122)
# plt.title('9*9卷积图像')
# plt.imshow(img5, cmap=plt.cm.gray)
#
# plt.show()
#
# from skimage import data, color
# from matplotlib import pyplot as plt
# import numpy as np
#
# # 定义灰度级到彩色变换
# L = 255
#
#
# def GetR(gray):
#     if gray < L / 2:
#         return 0
#     elif gray > L / 4 * 3:
#         return L
#     else:
#         return 4 * gray - 2 * L
#
#
# def GetG(gray):
#     if gray < L / 4:
#         return 4 * gray
#     elif gray > L / 4 * 3:
#         return 4 * L - 4 * gray
#     else:
#         return L
#
#
# def GetB(gray):
#     if gray < L / 4:
#         return L
#     elif gray > L / 2:
#         return 0
#     else:
#         return 2 * L - 4 * gray
#
#
# img = plt.imread('D:\\pycharmproject\\20180663张清辉_python\\course\\1.jpg')
# grayimg = color.rgb2gray(img) * 255  # 将彩色图像转化为灰度图像
# colorimg = np.zeros(img.shape, dtype='uint8')
# for ii in range(img.shape[0]):
#     for jj in range(img.shape[1]):
#         r, g, b = GetR(grayimg[ii, jj]), GetG(grayimg[ii, jj]), GetB(grayimg[ii, jj])
#         colorimg[ii, jj, :] = (r, g, b)
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# plt.subplot(1, 3, 1)
# plt.axis('off')
# plt.imshow(img)
# plt.title('原图像')
# plt.subplot(1, 3, 2)
# plt.axis('off')
# plt.imshow(grayimg, cmap='gray')
# plt.title('灰度图像')
# plt.subplot(1, 3, 3)
# plt.axis('off')
# plt.imshow(colorimg)
# plt.title('伪彩色图像')
#
# plt.savefig('ruanyi.tif')



