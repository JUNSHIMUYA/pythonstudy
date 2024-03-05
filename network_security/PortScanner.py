#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@Author       :Qinghui Zhang
@Version      : V1.0
@E-Mail       : zhangqinghui@iie.ac.cn
@File         : PortScanner.py
@CreateTime   : 2023/1/1
@ModifyTime   : 2023/1/1
@company      : UCAS
"""



import nmap

nm=nmap.PortScanner()
nm.scan('192.168.1.101','1-500','-O')