#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/1 8:31 下午
# @Author  : Andy
# @Site    : Beijing
# @File    : tensor_data_type.py
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
import torch

'''
功能描述：


'''


def tensor_type():
    t_arr1 = torch.zeros((2, 3))
    logger.info("t_arr1 dtype = {}".format(t_arr1.dtype))  # torch.float32
    t_arr2 = torch.ones((2, 3))
    logger.info("t_arr2 dtype = {}".format(t_arr2.dtype))  # torch.float32
    t_arr3 = torch.ones((2, 3)).long()
    logger.info("t_arr3 dtype = {}".format(t_arr3.dtype))  # torch.int64


def main(argv):
    try:
        # main code
        logging_base_setting1()

        tensor_type()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
