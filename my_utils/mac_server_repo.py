#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/1/25 10:26 下午
# @Author  : Andy
# @Site    : Beijing
# @File    : mac_server_repo.py
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
from my_proj_config.my_proj_config import IS_MAC

# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - |%(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)


'''
功能描述：  
/home/yangyang.andy/code   <-->   /Volumes/home1/code
'''
def path_auto_transform(path:str):
    # 需要判断两个因素： 1. 当前运行环境是在MAC上，还是在服务器上  2. path的地址是在MAC上的，还是在服务器上的

    str_prefix_on_mac = "/Volumes/home1"
    str_prefix_on_server = "/home/yangyang.andy"



    def path_is_on_mac(path:str):
        return path.startswith(str_prefix_on_mac)

    def path_is_on_server(path:str):
        return path.startswith(str_prefix_on_server)

    # 判断运行环境
    b_running_on_mac = True if IS_MAC else False

    # 判断path的地址是在MAC上的，还是在服务器上的
    if path_is_on_mac(path):
        b_path_on_mac = True
    elif path_is_on_server(path):
        b_path_on_mac = False
    else:
        raise Exception("path = {}. can not tell on mac or server.".format(path))


    if b_running_on_mac:
        if b_path_on_mac:
            pass
        else:
            return path.replace(str_prefix_on_server, str_prefix_on_mac)

    else:
        if b_path_on_mac:
            return path.replace(str_prefix_on_mac, str_prefix_on_server)
        else:
            return path








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
