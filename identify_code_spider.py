#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2018/05/02
# @Author  : kingsley kwong
# @File    : shop\api.py
# @Software: BI 不漏 flask


import requests
from bs4 import BeautifulSoup

for i in range(10):
    r = requests.get('https://vis.vip.com/checkCode.php?t=0.824002121384285')
    with open(f'E:\唯品会验证码\{i}.png', 'wb') as img:
        img.write(r.content)

