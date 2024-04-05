import logging
import sys
import os
from my_proj_config.my_proj_config import PROJ_ROOT
logger = logging.getLogger(__name__)





def logging_base_setting1(write_mode='w'):
    root_set()
    root_add_sysstd()
    root_add_file(write_mode)





def root_set():
    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)



def root_add_sysstd():
    root_logger = logging.getLogger()
    myconsole_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - |%(message)s')
    myconsole_handler.setFormatter(formatter)
    myconsole_handler.setLevel(logging.INFO)
    root_logger.addHandler(myconsole_handler)

    return myconsole_handler





def root_add_file(write_mode='w'):
    root_logger = logging.getLogger()
    myfile_handler = logging.FileHandler(os.path.join(PROJ_ROOT, "mylog.log"), mode=write_mode)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - |%(message)s')
    myfile_handler.setFormatter(formatter)
    myfile_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(myfile_handler)

    return myfile_handler

