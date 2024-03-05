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
from turtle import *       # 导入库


pensize(2)                 # 设置画笔大小
pencolor("red")            # 设置画笔颜色
speed(1)                   # 设置速度

penup()                    # 提起画笔
goto(-210,210)             # 画笔去指导位置
right(90)
pendown()
forward(160)

penup()
goto(-200,170)
pendown()
write("我", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-130,170)
pendown()
write("0", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-70,170)
pendown()
write("生", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-200,120)
pendown()
write("0", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-140,120)
pendown()
write("有", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-70,120)
pendown()
write("0", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-200,60)
pendown()
write("你", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-130,60)
pendown()
write("0", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-70,60)
pendown()
write("幸", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(-10,210)
pendown()
forward(160)

penup()
goto(0,130)
left(90)
pendown()
forward(40)

penup()
goto(0,110)
pendown()
forward(40)

penup()
goto(50,100)
pendown()
write("我有幸一生有你", font=('Arial', 30, 'normal'))  # 写字

penup()
goto(0,-60)
pendown()

color('red', 'pink')  # 画笔色red，背景色pink
begin_fill()
left(135)  # 左转135°
forward(100)  # 前进100像素
right(180)  # 画笔掉头
circle(30, -180)
backward(35)  # 由于此时画笔方向约为绝对方向的135°，需倒退画线
right(90)
forward(35)
circle(-30, 180)
fd(100)
end_fill()
hideturtle()

done()



