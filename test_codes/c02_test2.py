#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 下午7:32
# @Author  : Andy
# @Site    : 
# @File    : c02_test2.py
# @Software: PyCharm

import os
import sys
import numpy as np
import json
import pickle
import shutil
import logging

import torch as t

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '.')), ''))

from my_logging.my_logging2 import logging_base_setting1

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - |%(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)


'''
功能描述： 


'''


class MyClass(object):
    def __init__(self):
        pass

    def main_proc(self):
        pass


def main(argv):
    try:
        # main code 
        logging_base_setting1()

        x = t.ones(2, 2, requires_grad=True)
        y = x.sum()
        print(y.grad_fn)
        print(x.grad)
        y.backward()
        print(x.grad)
        y.backward()
        print(x.grad)


        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
