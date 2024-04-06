#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/14 下午4:00
# @Author  : Andy
# @Site    : 
# @File    : manual_seed_c04.py
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
'''
功能描述： 
是被  seed_test1_c04.py 调用的， 目前没有其他作用
'''

def create_tensor2():
    a2 = t.rand(3, 4)
    print("a2 = {}".format(a2))


def main(argv):
    try:
        # main code 
        logging_base_setting1()
        t.manual_seed(2)
        create_tensor2()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
