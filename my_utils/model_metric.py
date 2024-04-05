#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 6:40 下午
# @Author  : Andy
# @Site    : Beijing
# @File    : my_metrics.py
# @Software: PyCharm

import os
import sys
import numpy as np
import json
import pickle
import shutil
import logging

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '.')), ''))

from my_logging.my_logging2 import logging_base_setting1

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - |%(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)


'''
功能描述：


'''

class SamplePredGtInfo(object):
    def __init__(self):
        self.sample_name = None
        self.f_pred = None
        self.i_pred = None
        self.i_gt = None


class ModelMetric(object):
    def __init__(self):

        self.total_sample = 0
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

        # 模型命中量
        self.hit = 0

        self.reset()

        self.precision = None
        self.recall = None
        self.accuracy = None


    def reset(self):
        self.total_sample = 0
        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

        # 模型命中量
        self.hit = 0


    def add_true_positive(self):
        self.true_positive += 1
        self.total_sample += 1
        self.hit += 1


    def add_false_positive(self):
        self.false_positive += 1
        self.total_sample += 1
        self.hit += 1

    def add_true_negative(self):
        self.true_negative += 1
        self.total_sample += 1

    def add_false_negative(self):
        self.false_negative += 1
        self.total_sample += 1


    # 记录每个case的结果
    def pred_detail_info(self, sample_info:SamplePredGtInfo):
        pass




    def print_data(self):
        logger.info("total_sample = {}".format(self.total_sample))
        logger.info("true_positive = {}".format(self.true_positive))
        logger.info("false_positive = {}".format(self.false_positive))
        logger.info("true_negative = {}".format(self.true_negative))
        logger.info("false_negative = {}".format(self.false_negative))
        logger.info("hit = {}".format(self.hit))

    def cal_precision(self):
        if self.true_positive+self.false_positive == 0:
            self.precision = -1
        else:
            self.precision = (self.true_positive)/float(self.true_positive+self.false_positive)

    def cal_recall(self):
        if self.true_positive+self.false_negative == 0:
            self.recall = -1.0
        else:
            self.recall = (self.true_positive)/float(self.true_positive+self.false_negative)

    def cal_accuracy(self):
        if self.total_sample == 0:
            self.accuracy = -1.0
        else:
            self.accuracy = (self.true_positive+ self.true_negative)/float(self.total_sample)

    # 输出模型指标
    def print_index(self):
        logger.info("precision = {}".format(self.precision))
        logger.info("recall = {}".format(self.recall))
        logger.info("accuracy = {}".format(self.accuracy))

    # 计算漏放率
    def cal_loufang(self):
        pass



    def main_proc(self):
        pass


def main(argv):
    try:
        # main code
        logging_base_setting1()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
