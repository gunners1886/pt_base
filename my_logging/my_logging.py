#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/18 10:34 AM
# @Author  : Andy
# @Site    : 
# @File    : mylogging.py
# @Software: PyCharm

import os
import sys
import numpy as np
import json
import pickle
import shutil
import logging

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '..')), ''))

'''
功能描述： 


'''


from my_proj_config.my_proj_config import PROJ_ROOT

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.DEBUG)

myfile_handler = logging.FileHandler(os.path.join(PROJ_ROOT, "mylog.log"),mode='w')
myfile_handler.setLevel(logging.INFO)   # 取 logger 和 myfile_handler 的level的最高level作为最终的level

myconsole_handler = logging.StreamHandler()
myconsole_handler.setLevel(logging.DEBUG)  # 取 logger 和 myconsole_handler 的level的最高level作为最终的level



# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - |%(message)s')
myfile_handler.setFormatter(formatter)
myconsole_handler.setFormatter(formatter)


logger.addHandler(myfile_handler)
logger.addHandler(myconsole_handler)


myfile_handler_list = []

def remove_handler(logger_, handler_):
    logger_.removeHandler(handler_)


def add_file_handler(logger_):
    myfile_handler_new = logging.FileHandler(os.path.join(PROJ_ROOT, "change.log"), mode='w')
    myfile_handler_new.setLevel(logging.INFO)  # 取 logger 和 myfile_handler 的level的最高level作为最终的level

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    myfile_handler_new.setFormatter(formatter)
    logger_.addHandler(myfile_handler_new)



# 'FileHandler'    'StreamHandler'
def get_handler_type(handler_):
    return handler_.__class__.__name__


