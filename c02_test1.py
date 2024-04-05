#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/8 下午5:28
# @Author  : Andy
# @Site    : 
# @File    : c02_test1.py
# @Software: PyCharm

from __future__ import print_function
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

问题：
1. 什么时候使用 clone() 和 detach()?  如何理解slowfast代码中的clone和detach方法呢？



'''



import torch as t



class MyClass(object):
    def __init__(self):
        pass

    def main_proc(self):
        pass


def main(argv):
    try:
        # main code 
        logging_base_setting1()
        x = t.rand(5, 3)

        # size() 和 shape 是一样的 https://github.com/pytorch/pytorch/issues/5544
        print(x.shape)
        print(x.size())

        tensor = t.tensor([3,4])
        old_tensor = tensor  # 这个 = 操作不会新分配内存
        old_tensor[0] = 1111
        print(tensor)
        print(old_tensor)

        new_tensor = old_tensor.detach()
        new_tensor[0] = 2222
        print(old_tensor)
        print(new_tensor)

        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        x = t.rand(5, 3)
        x = x.to(device)
        print("11111")

        y = t.rand(5, 3)
        y = y.to(x.device)
        print("22222")
        z = x+y





        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
