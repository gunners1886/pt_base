#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/14 下午3:38
# @Author  : Andy
# @Site    : 
# @File    : c04_neural_network_test1.py
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
import torch as t
from torch import nn

import torch.utils.model_zoo as model_zoo

'''
功能描述： 


'''


class Linear(nn.Module): # 继承nn.Module
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__() # 等价于nn.Module.__init__(self)
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w) # x.@(self.w)
        return x + self.b.expand_as(x)


def main(argv):
    try:
        # main code 
        logging_base_setting1()

        linear = nn.Linear(3, 4)
        print("type of linear = {}".format(type(linear.weight))) # <class 'torch.nn.parameter.Parameter'>

        conv1 = nn.Conv2d(512, 512, (3, 3))
        print("type of conv1 = {}".format(type(conv1.weight)))

        from PIL import Image


        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
