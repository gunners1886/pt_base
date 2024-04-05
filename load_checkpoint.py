#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/2/21 4:01 下午
# @Author  : Andy
# @Site    : Beijing
# @File    : load_checkpoint.py
# @Software: PyCharm

import os
import sys
import numpy as np
import json
import pickle
import shutil
import logging
import torch

if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '.')), ''))

from my_logging.my_logging2 import logging_base_setting1

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - |%(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)
from my_utils.mac_server_repo import path_auto_transform
from my_proj_config.my_proj_config import IS_MAC

'''
功能描述：
pyroch自动生成的pth文件中，都包含哪些信息？
'''


def  laod_checkpoint(str_path):
    if IS_MAC:
        return torch.load(str_path, map_location=torch.device('cpu'))
    else:
        return torch.load(str_path, map_location=torch.device('cpu'))





def main(argv):
    try:
        # main code
        logging_base_setting1()

        str_model_path = "/home/yangyang.andy/hdfs/pangolin/train/try_0005/step04_model/epoch_001.pth"
        str_model_path = path_auto_transform(str_model_path)
        checkpoint = laod_checkpoint(str_model_path)



        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
