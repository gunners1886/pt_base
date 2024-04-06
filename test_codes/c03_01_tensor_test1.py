#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/12 上午9:59
# @Author  : Andy
# @Site    : 
# @File    : c03_test1.py
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
# 默认类型是 torch.float32 吗？
def default_dtype():
    a = t.Tensor(2, 3)
    print(a.dtype)

    b = t.ones(2, 3)
    print(b.dtype)

    c = t.zeros(2, 3)
    print(c.dtype)

    print(t.linspace(1, 10, 3).dtype) # torch.float32
    print(t.randn(2, 3, device=t.device('cpu')).dtype) # torch.float32

    ## 例外情况
    print(t.arange(1, 6, 2).dtype)  # torch.int64
    print(t.tensor([2, 3]).dtype)   # torch.int64


# 参考 https://github.com/chenyuntc/pytorch-book/blob/master/chapter03-tensor_and_autograd/Tensor.ipynb
# 表3-1: 常见新建tensor的方法   !!!!!!!
def create_tensor():
    # [0, 1]之间的 均匀 分布；  rand可以认为是随机的意思
    a = t.rand(2, 3)
    print("a = {}".format(a))
    print("a.dtype = {}".format(a.dtype)) # torch.float32

    # 标准正态分布；  randn中的n可以认为是normal的意思
    b = t.randn(2, 3)
    print(b)
    print(b.dtype)  # torch.float32



def input_is_size():
    a = t.Tensor(2, 3)
    print(a)

    b = t.Tensor([2, 3])
    print(b)       # tensor([2., 3.])
    print(b.dtype) # torch.float32

    c = t.tensor([2, 3])
    print(c) #tensor([2, 3])
    print(c.dtype) # torch.int64



def test():
    a = t.randn(3, 4)
    print(a)
    print(a[t.LongTensor([0,1])])

    # 下面的语句会报错！！
    # print(a[t.Tensor([0,1])]) # tensors used as indices must be long, byte or bool tensors

# 结论： 除非是te除非是tensor.function_(), 一般的tensor.funcion()都不改变tensor本身，如果需要结果，那么要用一个 ret 承接返回结果
def tensor_function_ret():
    a = t.randn(3, 4)
    print("a = {}".format(a))

    ret_b = a.view(4, 3)
    print(a.shape)  # torch.Size([3, 4]), a的形状没有变
    print(ret_b.shape)  # torch.Size([4, 3]), ret_b 的形状相对于a改变了的

    ret_c = a.reshape(4, 3)
    print(a.shape)  # torch.Size([3, 4]), a的形状没有变
    print(ret_c.shape)  # torch.Size([4, 3]), ret_c 的形状相对于a改变了的


    b = t.randn(3, 4)
    c = a.add(b)
    print("add 1 = {}".format(a.add(b)))
    print("a = {}".format(a))
    print("c = {}".format(c))

# 结论： tensor.shape = tensor.size()
def shape_size():
    a = t.randn(3, 4)
    print(a.shape) # torch.Size([3, 4])
    print(a.size()) # torch.Size([3, 4])


# 使用view 还是 reshape?  结论： 优先选择reshape, 偶尔当确定内存是连续的，可以使用view, 最好不用resize_
# 基本用法相同： https://blog.csdn.net/Flag_ing/article/details/109129752
# 总结： view能做的reshape都看可以，reshape能处理的， view不一定能。 因此无脑情况下，使用reshape()
# 与resize的区别： 69/301   深度学习框架PyTorch：入门与实践_陈云(著)  .pdf
def view_reshape():
    a = t.randn(3, 4)
    print(a)

    b = a.view(4, 3) # 这里由于 tensor_function_ret 已经验证， 不会改变a，因此a可以继续用
    print(b.shape) # torch.Size([4, 3])
    c = a.reshape(4, 3)  # 这里由于 tensor_function_ret 已经验证， 不会改变a，因此a可以继续用
    print(c.shape) # torch.Size([4, 3])

    # d = a.view(5, 3)  # RuntimeError: shape '[3, 3]' is invalid for input of size 12
    # e = a.reshape(5, 3) # RuntimeError: shape '[3, 3]' is invalid for input of size 12



def main(argv):
    try:
        # main code
        logging_base_setting1()

        # 默认类型是 torch.float32 吗？
        # default_dtype()
        # create_tensor()
        # input_is_size()

        # 结论: 除非是tensor.function_(), 一般的tensor.funcion()都不改变tensor本身，如果需要结果，那么要用一个 ret 承接返回结果
        tensor_function_ret()
        # 结论： tensor.shape = tensor.size()
        # shape_size()

        # 结使用view 还是 reshape?  结论： 优先选择reshape, 偶尔当确定内存是连续的，可以使用view, 最好不用resize_
        # view_reshape()


        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
