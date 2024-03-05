#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@Author       : 20180663 Zhang Qinghui
@Version      : V1.0
@E-Mail       : 1415984778@qq.com
@File         : test6.py
@CreateTime   : 2021/4/15
@Description  : spider
@ModifyTime   : 2021/4/15
@company      : CSUFT
"""

from bs4 import BeautifulSoup
import requests


def SG():

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36'
    }
    url = 'https://www.shicimingju.com/book/sanguoyanyi.html'

    page = requests.get(url=url, headers=headers)
    page.encoding = "utf-8"
    soup = BeautifulSoup(page.text, 'lxml')
    a_list = soup.select('.book-mulu > ul >li >a ')
    for a in a_list:
        title = a.text
        detail_url = "https://www.shicimingju.com"+a['href']
        context = requests.get(url=detail_url, headers=headers)
        context.encoding = "utf=8"
        soup = BeautifulSoup(context.text, 'lxml')
        detail_context = soup.find('div', class_='chapter_content').text
        filepath = './三国演义/'+title+'.txt'
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(title+'\n')
            f.write(detail_context)
    print("成功")


if __name__ == "__main__":

    SG()
