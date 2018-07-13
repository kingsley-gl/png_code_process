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
# for pic in os.listdir('E:/vip_png/'):
# 灰度二值化处理
img = cv2.imread('E:/vip_png/0.png')
blur = cv2.blur(img, (4, 2))  # 验证码模糊
img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 验证码灰度
ret, img_twovalue = cv2.threshold(img_gray, thresh=191, maxval=255, type=cv2.THRESH_BINARY)  # 验证码二值化
# img_digit_list.append(img_twovalue)
print(len(img_twovalue))


# 投影法分割字符
def projection(image_digit):
    white = []
    black = []
    height = image_digit.shape[0]
    width = image_digit.shape[1]
    print(height, width)
    white_max = 0
    black_max = 0

    # 统计每一列的黑白色像素总和
    for i in range(width):
        s = 0  # 列白色总和
        t = 0  # 列黑色总和
        for j in range(height):
            if image_digit[j][i] == 255:
                s += 1
            if image_digit[j][i] == 0:
                t += 1

        white_max = max(white_max, s)
        black_max = max(black_max, t)
        white.append(s)
        black.append(t)

    arg = False  # False表示白底黑字；True表示黑底白字
    if black_max > white_max:
        arg = True

    def find_end(start_):
        end_ = start_ + 1
        for m in range(start_ + 1, width - 1):
            if (black[m] if arg else white[m]) > (0.90 * black_max if arg else 0.90 * white_max):
                end_ = m
                break
        return end_

    n = 1
    start = 1
    end = 2

    while n < width - 2:
        n += 1
        if (white[n] if arg else black[n]) > (0.10 * white_max if arg else 0.10 * black_max):
            # 上面这些判断用来辨别是白底黑字还是黑底白字
            # 0.05这个参数请多调整，对应上面的0.95
            start = n
            end = find_end(start)
            n = end
            if end - start > 5:
                cj = image_digit[1:height, start:end]
                cv2.imshow('caijian', cj)
                cv2.waitKey(0)


class drop_fall(object):
    def __init__(self, image_digit):
        self.image_digit = image_digit
        self.height = image_digit.shape[0]
        self.width = image_digit.shape[1]

    def vertical(self):
        '''
        垂直投影
        :return:
        '''
        result = []
        for x in range(self.height):
            black = 0
            for y in range(self.width):
                if self.image_digit[x][y] == 0:
                    black += 1
            result.append(black)
        return result

    def get_start_x(self, hist_width):
        '''
        根据垂直投影确定起点
        :param hist_width:
        :return:
        '''
        mid = len(hist_width)
        temp = hist_width[mid - 4:mid + 5]
        return mid - 4 + temp.index(min(temp))

    def get_nearby_pix_value(self, x, y, j):
        '''
        获取临近5个点的像素数据
        :param x:
        :param y:
        :param j:
        :return:
        '''
        if j == 1:
            return 0 if self.image_digit[x - 1][y + 1] == 0 else 1
        elif j == 2:
            return 0 if self.image_digit[x][y + 1] == 0 else 1
        elif j == 3:
            return 0 if self.image_digit[x + 1][y + 1] == 0 else 1
        elif j == 4:
            return 0 if self.image_digit[x + 1][y] == 0 else 1
        elif j == 5:
            return 0 if self.image_digit[x - 1][y] == 0 else 1
        else:
            raise Exception("get_nearby_pix_vallule error")

    def get_end_route(self, start_x, height):
        '''
        获取水滴路径
        :param start_x:
        :param height:
        :return:
        '''
        left_limit = 0
        right_limit = self.width - 1
        end_route = []
        cur_p = (start_x, 0)
        last_p = cur_p
        end_route.append(cur_p)

        while cur_p[1] < (height - 1):
            sum_n = 0
            max_w = 0
            next_x = cur_p[0]
            next_y = cur_p[1]
            for i in range(1, 6):
                cur_w = self.get_nearby_pix_value(cur_p[0], cur_p[1], i) * (6 - i)
                sum_n += cur_w
                if max_w < cur_w:
                    max_w = cur_w
            if sum_n == 0:
                # 全黑
                max_w = 4
            if sum_n == 15:
                # 全白
                max_w = 6

            if max_w == 1:
                next_x = cur_p[0] - 1
                next_y = cur_p[1]
            elif max_w == 2:
                next_x = cur_p[0] + 1
                next_y = cur_p[1]
            elif max_w == 3:
                next_x = cur_p[0] + 1
                next_y = cur_p[1] + 1
            elif max_w == 5:
                next_x = cur_p[0] - 1
                next_y = cur_p[1] + 1
            elif max_w == 6:
                next_x = cur_p[0]
                next_y = cur_p[1] + 1
            elif max_w == 4:
                if next_x > cur_p[0]:
                    # 向右
                    next_x = cur_p[0] + 1
                    next_y = cur_p[1] + 1
                if next_x < cur_p[0]:
                    next_x = cur_p[0]
                    next_y = cur_p[1] + 1
                if sum_n == 0:
                    next_x = cur_p[0]
                    next_y = cur_p[1] + 1
            else:
                raise Exception("get end route error")

            if last_p[0] == next_x and last_p[1] == next_y:
                if next_x < cur_p[0]:
                    max_w = 5
                    next_x = cur_p[0] + 1
                    next_y = cur_p[1] + 1
                else:
                    max_w = 3
                    next_x = cur_p[0] - 1
                    next_y = cur_p[1] + 1
                last_p = cur_p

            if next_x > right_limit:
                next_x = right_limit
                next_y = cur_p[1] + 1

            if next_x < left_limit:
                next_x = left_limit
                next_y = cur_p[1] + 1

            cur_p = (next_x, next_y)
            end_route.append(cur_p)
        return end_route

    def get_split_seq(self, projection_x):
        split_seq = []
        start_x = 0
        length = 0
        for pos_x, val in enumerate(projection_x):
            if val == 0 and length == 0:
                continue
            elif val == 0 and length != 0:
                split_seq.append([start_x, length])
                length = 0
            elif val == 1:
                if length == 0:
                    start_x = pos_x
                length += 1
            else:
                raise Exception('generating split sequence occurs error')

        if length != 0:
            split_seq.append([start_x, length])
        return split_seq

    def do_split(self, starts, filter_ends):
        left = starts[0][0]
        top = starts[0][1]
        right = filter_ends[0][0]
        bottom = filter_ends[0][1]
        pixdata = self.image_digit
        for i in range(len(starts)):
            left = min(starts[i][0], left)
            top = min(starts[i][0], top)
            right = min(starts[i][0], right)
            bottom = min(starts[i][0], bottom)
        width = right - left + 1
        height = bottom - top + 1


# projection(img_twovalue)
plt.imshow(img_twovalue, 'gray')
plt.show()

# https://www.jb51.net/article/141461.htm

# cv2.waitKey(0)
# print(img_digit_list)
# img_data = datasets.fetch_20newsgroups(img_digit_list)
# print(img_data)
#
# Xtrain, Xtest, ytrain, ytest = train_test_split(digits.data, digits.target, random_state=2)
#
# clf = LogisticRegression()
# clf.fit(Xtrain, ytrain)
# ypred = clf.predict(Xtest)
# accuracy = accuracy_score(ytest, ypred)
# print("识别准确度：", accuracy)
