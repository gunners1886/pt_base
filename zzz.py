#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/5/26 4:47 下午
# @Author  : Andy
# @Site    : Beijing
# @File    : zzz.py
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

        from torchvision import transforms
        np_image = np.random.random([5, 5, 3])
        np_image = transforms.functional.to_tensor(np_image)

        logger.info(np_image)

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
