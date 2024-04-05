#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/15 8:31 下午
# @Author  : Andy
# @Site    : Beijing
# @File    : tensor_to_numpu.py
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

def tensor_to_numpy():
    # [0, 1]之间的 均匀 分布；  rand可以认为是随机的意思
    ts_data = torch.rand(2, 3)
    print("ts_data = {}".format(ts_data))
    print("ts_data.dtype = {}".format(ts_data.dtype))  # torch.float32

    np_data = ts_data.numpy()

    print(np_data)


def numpy_to_tensor():
    data = [[1, 2, 3], [4, 5, 7]]
    np_array = np.array(data)
    ts_data = torch.from_numpy(np_array)

    print("type is {}".format(type(ts_data)))



def main(argv):
    try:
        # main code
        logging_base_setting1()

        # tensor_to_numpy()
        numpy_to_tensor()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
