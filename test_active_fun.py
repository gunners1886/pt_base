#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/9/29 10:42
# @Author  : Andy
# @Site    : Beijing
# @File    : test_active_fun.py
# @Software: PyCharm

import os
import sys
import numpy as np
import json
import pickle
import shutil
import logging
import yaml

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '.')), ''))

from my_logging.my_logging2 import logging_base_setting1

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - |%(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)
from my_proj_config.my_proj_config import PROJ_ROOT
import torch
import torch.nn.functional as F

def sigmoid_fun():
    l1 = [1, 2, 3, 4]
    l2 = [1, 1, 1, 1]
    l3 = [10000, 1, 2, 4]

    np_arr = np.array([l1, l2, l3], dtype=float)

    output = torch.tensor(np_arr)
    prob = F.softmax(output, dim=1)

    logger.info("output_shape = {}".format(output.shape))
    logger.info("prob_shape = {}".format(prob.shape))
    logger.info("prob = {}".format(prob))


def main(argv):
    try:
        # main code
        logging_base_setting1()

        sigmoid_fun()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
