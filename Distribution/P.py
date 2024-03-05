
#!/usr/bin/python

'''
-*- coding: utf-8 -*-
@File  : p.py
@school: UCAS
@Time  : 2022/08/15 22:33
'''


import socket


N = 20            # 消息轮转次数
msg = 'm'           # 轮转消息
counts = 88         # 初始状态
marker='s'          # snapshot 开始标志
record=[]           # 存储快照记录

def snapshot(i):    # 快照记录函数
    item={"p_m":i}
    record.append(item)


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("127.0.0.1", 54321))
sock.listen(5)
connection, address = sock.accept()

while N:            # 开始消息轮转

    N = N - 1;
    connection.send(msg.encode())           # 进程p先发送消息
    buf = connection.recv(1024).decode()    # 进程p接收来自进程q的消息
    counts = counts + 1                     # 接收到消息m后状态+1
    if counts==101:
        connection.send(marker.encode())     # 状态=101时 发送marker告诉进程q启动快照
        print("p start snapshot")             # 启动快照算法
        snapshot(counts)                      # 记录快照
    if counts>101:                             # 快照算法开启后每次都记录状态并且打印
        snapshot(counts)
        print(record)
    print("p has gotten {} times msg and the msg is {}\n".format(counts, buf))

connection.close()