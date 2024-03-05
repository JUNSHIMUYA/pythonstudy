#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@Author       : 20180663 Zhang Qinghui
@Version      : V1.0
@E-Mail       : 1415984778@qq.com
@File         : test4.py
@CreateTime   : 2021/3/20
@Description  : A math problem
@ModifyTime   : 2021/3/20
@company      : CSUFT
"""
# 用牛顿迭代法计算一个数的立方根
import math                                # 导入数学计算库


a = float(input("请输入一个数："))          # 输入一个数


def f1(x):                                 # 定义函数 f(x)=x^3-a (x^3代表x的三次方)
    return x**3-a


def f2(x):                                 # 函数f1(x)的导数
    return 3*x**2


ep = 0.00000001                            # 定义精度
xk = 1                                     # 迭代初始值
xk1 = 1

while True:
    xk1 = xk                               # 开始迭代
    xk = xk1-f1(xk1)/f2(xk1)
    if math.fabs(xk-xk1) < ep:
        break
print("{}的立方根为{}".format(a,xk))
