#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@Author       : 20180663 Zhang Qinghui
@Version      : V1.0
@E-Mail       : 1415984778@qq.com
@File         : test5.py
@CreateTime   : 2021/3/25
@Description  : the using of matplotlib
@ModifyTime   : 2021/3/25
@company      : CSUFT
"""

import matplotlib.pyplot as plt
import numpy as np

x, y = np.mgrid[-2:2:20j, -2:2:20j]
z = 50*np.sin(x+y)                                                   # 测试数据
ax = plt.subplot(111, projection='3d')                               # 三维图形
ax.plot_surface(x, y, z, rstride=2, cstride=1, cmap=plt.cm.Blues_r)
ax.set_xlabel('X')                                                   # 设置坐标轴标题
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

