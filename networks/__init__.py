#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 下午7:07
# @Author  : Andy
# @Site    : 
# @File    : __init__.py.py
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

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
