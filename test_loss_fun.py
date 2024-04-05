#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 09:45
# @Author  : Andy
# @Site    : Beijing
# @File    : loss_fun.py
# @Software: PyCharm

import os
import sys
import numpy as np
import json
import pickle
import shutil
import logging
import yaml
import torch
import torch.nn as nn

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '.')), ''))

from my_logging.my_logging2 import logging_base_setting1

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - |%(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)
from my_proj_config.my_proj_config import PROJ_ROOT


def nn_bce_loss():
    '''
    BCE loss
    代码来源： https://www.jianshu.com/p/5b01705368bb
    input: 4x3  4=batch_size, 3=dim_num
    target: 4x3  4=batch_size, 3=dim_num     ---  这个和ce有明显区别！！
    :return:
    '''
    sig = nn.Sigmoid()
    bce = nn.BCELoss()
    input = torch.randn((4, 3), requires_grad=True, dtype=float)

    # target = torch.tensor(np.array([1, 1, 0], dtype=float))
    target = torch.tensor(np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=float))

    output = bce(sig(input), target)
    output.backward()

    logger.info('input_shape = {}'.format(input.shape))
    logger.info('target_shape = {}'.format(target.shape))

    logger.info('input = {}'.format(input))
    logger.info('target = {}'.format(target))

    # logger.info('sig(input)', sig(input))
    logger.info("output_shape = {}".format(output.shape))
    logger.info("output = {}".format(output))




def ce_loss():
    '''
    CE loss

    input: 4x3  4=batch_size, 3=dim_num
    target: 4  4=batch_size   ---  这个和bce有明显区别！！


    :return:
    '''

    import torch.nn.functional as F

    ce = nn.CrossEntropyLoss()
    input = torch.randn((4, 3), requires_grad=True)
    prob = F.softmax(input, dim=1)
    target = torch.tensor(np.array([1, 1, 1, 0]))
    # target = torch.tensor(np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]))

    output = ce(prob, target)
    output.backward()

    logger.info('input_shape = {}'.format(input.shape))
    logger.info('target_shape = {}'.format(target.shape))

    logger.info('input = {}'.format(input))
    logger.info('target = {}'.format(target))

    # logger.info('sig(input)', sig(input))
    logger.info("output_shape = {}".format(output.shape))
    logger.info("output = {}".format(output))






def main(argv):
    try:
        # main code
        logging_base_setting1()
        
        # BCE loss
        nn_bce_loss()

        # CE loss
        # ce_loss()


        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
