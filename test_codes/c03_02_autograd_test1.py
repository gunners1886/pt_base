#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/12 下午5:17
# @Author  : Andy
# @Site    : 
# @File    : c03_02_autograd_test1.py
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
from torch.autograd import Variable as V
'''
功能描述： 
1. Tensor 与 Variable 的不同?   0.4版本后合并了!

'''

def test1():

    a = t.randn(3, 4, requires_grad=True)
    b = t.zeros(3, 4, requires_grad=True)
    c = t.add(a, b)
    d = c.sum()

    print(type(a))
    print(type(b))
    print(type(c))
    print(type(d))
    print(d)


    d.backward()

    print(a.grad)
    print(b.grad)

def test2():
    a = V(t.ones(3, 4), requires_grad=True)
    b = V(t.zeros(3, 4), requires_grad=True)

    c = t.add(a, b)
    d = c.sum()

    print(c.data.sum())
    print(c.sum())

    d.backward()

def main(argv):
    try:
        # main code 
        logging_base_setting1()

        # test1()
        test2()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
