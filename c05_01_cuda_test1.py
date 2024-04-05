#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/16 上午8:47
# @Author  : Andy
# @Site    : 
# @File    : c05_01_cuda_test1.py
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


'''


def test1():
    tensor0 = t.Tensor(3, 4)
    # 返回一个新的tensor，保存在第1块GPU上，但原来的tensor并没有改变
    tensor1 = tensor0.cuda(device=0)
    print(tensor0.is_cuda) # False
    print(tensor1.is_cuda) # True


def test2():
    # 交叉熵损失函数，带权重
    # criterion = t.nn.CrossEntropyLoss(weight=t.Tensor([1, 3]))
    criterion = t.nn.CrossEntropyLoss()
    input = t.randn(4, 2).cuda()
    print("type of input = {}".format(type(input)))
    print("input is cuda = {}".format(input.is_cuda))
    target = t.Tensor([1, 0, 0, 1]).long().cuda()
    print("type of target = {}".format(type(target)))
    print("target is cuda = {}".format(target.is_cuda))

    # 不加下面这行会报错，因weight未被转移至GPU
    # criterion.cuda()  # 如没有这一行就报错： RuntimeError: Expected object of device type cuda but got device type cpu for argument #3 'weight' in call to _thnn_nll_loss_forward
    loss = criterion(input, target)
    print("type of loss = {}".format(type(loss)))
    print("loss is cuda = {}".format(loss.is_cuda))
    print(criterion._buffers)



def main(argv):
    try:
        # main code 
        logging_base_setting1()

        a = t.ones(2, 3)
        b = t.ones(2, 3) * 2

        a = a.cuda(device=0)

        c = a + b
        print(c)




        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
