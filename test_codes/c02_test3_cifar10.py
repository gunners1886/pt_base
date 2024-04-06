#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/9 下午4:35
# @Author  : Andy
# @Site    : 
# @File    : c02_test3_cifar10.py
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
from my_proj_config.my_proj_config import PROJ_ROOT

from datetime import datetime

import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch.nn as nn
from networks.lenet_c2_v1 import LeNet
import time
'''
功能描述： 


'''


class MyPtTrain(object):
    def __init__(self):
        self._str_cur_time = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # 20190321_071833_212

        self._str_this_run_output_root = os.path.join(PROJ_ROOT, 'output', '')
        if not os.path.exists(self._str_this_run_output_root):
            os.makedirs(self._str_this_run_output_root)
        assert os.path.exists(self._str_this_run_output_root), "self._str_this_run_output_root = {} does not exist.".format(self._str_this_run_output_root)

        self._str_output_model_dir = os.path.join(self._str_this_run_output_root, 'model', '')
        if not os.path.exists(self._str_output_model_dir):
            os.makedirs(self._str_output_model_dir)
        assert os.path.exists(self._str_output_model_dir), "self._str_output_model_dir = {} does not exist.".format(self._str_output_model_dir)

        self._str_output_summary_dir = os.path.join(self._str_this_run_output_root, 'summary', '')
        if not os.path.exists(self._str_output_summary_dir):
            os.makedirs(self._str_output_summary_dir)
        assert os.path.exists(self._str_output_summary_dir), "self._str_output_summary_dir = {} does not exist.".format(self._str_output_summary_dir)

        self._str_output_pickle_dir = os.path.join(self._str_this_run_output_root, 'pkl', '')
        if not os.path.exists(self._str_output_pickle_dir):
            os.makedirs(self._str_output_pickle_dir)
        assert os.path.exists(self._str_output_pickle_dir), "self._str_output_pickle_dir = {} does not exist.".format(self._str_output_pickle_dir)


        self._str_data_dir = os.path.join(PROJ_ROOT, 'data', '')
        if not os.path.exists(self._str_data_dir):
            os.makedirs(self._str_data_dir)
        assert os.path.exists(self._str_data_dir), "self._str_data_dir = {} does not exist.".format(self._str_data_dir)



    def main_proc(self):
        #### prepare input data
        # 第一次运行程序torchvision会自动下载CIFAR-10数据集，
        # 大约100M，需花费一定的时间，
        # 如果已经下载有CIFAR-10，可通过root参数指定

        # 定义对数据的预处理
        transform = transforms.Compose([
            transforms.ToTensor(), # 转为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # 归一化
        ])

        # 训练集
        trainset = tv.datasets.CIFAR10(
            root=self._str_data_dir,
            train=True,
            download=True,
            transform=transform)

        trainloader = t.utils.data.DataLoader(
            trainset,
            batch_size=4,
            shuffle=True,
            num_workers=2)

        # 测试集
        testset = tv.datasets.CIFAR10(
            root=self._str_data_dir,
            train=False,
            download=True,
            transform=transform)

        testloader = t.utils.data.DataLoader(
            testset,
            batch_size=4,
            shuffle=False,
            num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



        # (data, label) = trainset[100]
        # print(classes[label])
        # show = ToPILImage()
        # show((data + 1) / 2).resize((100, 100))


        #### define network arch
        net = LeNet()



        #### define optim & loss
        from torch import optim
        criterion = nn.CrossEntropyLoss() # 交叉熵损失函数
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


        #### calc grad and update


        #### train & val many epochs
        t.set_num_threads(1) # 原始代码设置为8，但是在lenovo-ubuntu上会报错，最多设置为3
        for epoch in range(10):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # 输入数据
                inputs, labels = data

                # 梯度清零
                optimizer.zero_grad()

                # forward + backward
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

                # 更新参数
                optimizer.step()

                # 打印log信息
                # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
                running_loss += loss.item()
                if i % 2000 == 1999: # 每2000个batch打印一下训练状态
                    print('[%d, %5d] loss: %.3f' \
                          % (epoch+1, i+1, running_loss / 2000))
                    running_loss = 0.0


        t.save(net.state_dict(), 'net.pth')
        print('Finished Training')

        correct = 0 # 预测正确的图片数
        total = 0 # 总共的图片数

        # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
        device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        # device = t.device("cpu")

        net.to(device)
        t1 = time.time()
        with t.no_grad():
            for data in testloader:
                images, labels = data


                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                _, predicted = t.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        print('10000张测试集中的准确率为: %d %%' % (100 * correct / total))
        print("time_inference = {}".format(time.time()-t1))




def main(argv):
    try:
        # main code 
        logging_base_setting1()

        pt_train = MyPtTrain()
        pt_train.main_proc()

        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
