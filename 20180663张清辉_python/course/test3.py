#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@Author       : 20180663 Zhang Qinghui
@Version      : V1.0
@E-Mail       : 1415984778@qq.com
@File         : test3.py
@CreateTime   : 2021/3/18
@Description  : Introduce of a student
@ModifyTime   : 2021/3/18
@company      : CSUFT
"""

# 太简单了就不写注释了

name = input("你的姓名是：")
name_word = input("你字是：")
sex = input("你的性别是：")
age = input("你年龄是：")
address = input("你的家庭住址是：")
hobby = input("你的爱好是：")
print('大家好，我叫{}，字{}，为什么叫"慕雅"?因为"慕"代表追求，"雅"代表优雅,组合起来寓意着"追求优雅"，性别{}，年龄{}岁，'
      '家住{}，''爱好是{}'.format(name, name_word, sex, age, address, hobby))
