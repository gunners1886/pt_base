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
尝试 t.manual_seed(1), 验证该函数在同文件的相同函数内是否有效？     同文件的不同函数是否有效？    不同文件的函数是否也有效？ 

即： 需要回答一个问题： t.manual_seed 的有效域是多大？  答案： 全局！

'''

def create_tensor():
    a1 = t.rand(3, 4)
    print("a1 = {}".format(a1))


def main(argv):
    try:
        # main code 
        logging_base_setting1()
        t.manual_seed(2)

        # 结论： 能够对当前函数中的后续语句也产生固定的效果!
        a0 = t.rand(3, 4)
        print("a0 = {}".format(a0))

        # 结论： 能够对当前python文件中的其他函数也产生固定的效果!
        create_tensor()


        # 结论： 能够对其他python文件中的随机函数也产生固定的效果!
        from seed_test2_c04 import create_tensor2
        create_tensor2()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
