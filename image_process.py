#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2018/05/02
# @Author  : kingsley kwong
# @File    : 
# @Software: BI 不漏 flask

import numpy as np
import cv2

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.metrics import accuracy_score
import os

img_digit_list = []
for pic in os.listdir('E:/vip_png/'):
    img = cv2.imread('E:/vip_png/'+pic)
    blur = cv2.blur(img, (4, 2))
    img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, img_twovalue = cv2.threshold(img_gray, thresh=191, maxval=255, type=cv2.THRESH_BINARY)
    img_digit_list.append(img_twovalue)
# ret, img_twovalue = cv2.threshold(img_gray, thresh=145, maxval=255, type=cv2.THRESH_BINARY)

# for val in img_twovalue:
#     print(val)

# plt.imshow(img_twovalue, 'gray')
# plt.show()
# cv2.imshow('image', img_gray)

# cv2.waitKey(0)
print(img_digit_list)
img_data = datasets.fetch_20newsgroups(img_digit_list)
print(img_data)
#
# Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)
#
# clf = LogisticRegression()
# clf.fit(Xtrain, ytrain)
# ypred = clf.predict(Xtest)
# accuracy = accuracy_score(ytest, ypred)
# print("识别准确度：", accuracy)