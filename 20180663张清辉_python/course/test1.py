#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@Author       : 20180663 Zhang Qinghui
@Version      : V1.0
@E-Mail       : 1415984778@qq.com
@File         : test1.py
@CreateTime   : 2021/3/4
@Description  : none
@ModifyTime   : 2021/3/4
@company      : CSUFT
"""
from turtle import *  # 导入库


pensize(5)            # 设置画笔大小
pencolor("yellow")    # 设置画笔颜色
fillcolor("red")      # 设置填充颜色

begin_fill()          # 开始填充
for _ in range(5):    # 五角星五个角
    forward(200)      # 向前走200
    right(144)        # 向右转144度
end_fill()            # 结束填充

done()                # 结束




