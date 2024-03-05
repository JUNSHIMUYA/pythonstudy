
#!/usr/bin/python

'''
-*- coding: utf-8 -*-
@File  : q.py
@school: UCAS
@Time  : 2022/08/15 22:33
'''


import socket

sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sock.connect(("127.0.0.1",54321))

N=20                 # 消息轮转次数
counts=88              # 初始状态
s=0                    # s=0表示没有开启快照算法，s=1表示已经开启快照算法
marker='s'             # snapshot 开始标志
record=[]              # 存储快照记录

def snapshot(i):        # 快照记录函数
    item={"q_m":i}
    record.append(item)

while N:                                  # 开始消息轮转

    N=N-1
    buf=sock.recv(1024).decode()       # 进程q接收来自进程p的消息
    counts = counts + 1                # 接收到消息m后状态+1
    print("q has gotten {} times msg and the msg is {}\n".format(counts, buf))

    if s == 1:                          # s=1表示已经开启快照算法
        snapshot(counts)
        print(record)

    if buf=='s':                        # 表示进程q收到来自进程p的通知要开启快照算法
        s=1                              # 表示已经开启快照算法
        snapshot(counts)                # 记录快照
        print(record)
        sock.send(marker.encode())      # 向进程p发送marker
    else:
        sock.send(buf.encode())        # 向进程p发送消息m


sock.close()