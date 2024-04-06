#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/18 12:40 PM
# @Author  : Andy
# @Site    :
# @File    : proj_config.py
# @Software: PyCharm

import os
import sys
import numpy as np
import json
import pickle
import shutil
import logging
import platform


'''
功能描述： 


'''


PROJ_ROOT = os.path.join(os.path.abspath(os.path.join((os.path.dirname(__file__)), '..')), '')

# IS_UBUNTU = True if platform.linux_distribution()[0] == 'debian' else False

# https://stackoverflow.com/questions/8220108/how-do-i-check-the-operating-system-in-python
# from sys import platform
# IS_MAC = True if platform == "darwin" else False

IS_PYTHON_3 = (sys.version_info.major == 3)
