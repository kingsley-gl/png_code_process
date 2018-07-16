#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2018/05/02
# @Author  : kingsley kwong
# @File    : shop\api.py
# @Software: BI 不漏 flask


import requests
import time
import random
from bs4 import BeautifulSoup

for i in range(20):
    r = requests.get('https://vis.vip.com/checkCode.php?t=0.824002121384285')
    with open(f'E:\\vip_png\{i}.png', 'wb') as img:
        img.write(r.content)
    # time.sleep(random.randint(4,8))

