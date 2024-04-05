#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/3/14 下午10:42
# @Author  : Andy
# @Site    : 
# @File    : models.py
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
from torchvision import models
from PIL import Image

import torchvision.transforms as transforms
from dataset.imagenet.my_imagenet import MyImageNet
import torch
'''
功能描述： 
pytorch 官方实现的基础网络

'''

transform_list = [
    transforms.CenterCrop(224),
    transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std =[0.229, 0.224, 0.225])]
img_to_tensor = transforms.Compose(transform_list)




class OfficialModel(object):
    def __init__(self, str_src_img_path):
        assert os.path.exists(str_src_img_path), "str_src_img_path = {} does not exist.".format(str_src_img_path)
        self.str_src_img_path = os.path.abspath(str_src_img_path)

        self.model = None


    def get_vgg_model(self):
        # my_model = models.vgg16(pretrained=True, progress=True)
        # my_model = models.resnet34(pretrained=True, progress=True)
        # my_model = models.resnet50(pretrained=True, progress=True)
        # my_model = models.resnet101(pretrained=True, progress=True)
        my_model = models.resnet152(pretrained=True, progress=True)

        my_model.cuda()

        return my_model

    def get_data(self):
        pil_img = Image.open(self.str_src_img_path)

        width, height = pil_img.size

        # np_zero = np.zeros([height, width, 3], dtype=np.uint8)
        np_zero = np.ones([height, width, 3], dtype=np.uint8) * 255
        np_zero = np.ones([height, width, 3], dtype=np.uint8) * 128
        pil_img = Image.fromarray(np_zero)

        tensor_img = img_to_tensor(pil_img)

        logger.info("shape of tensor_img = {}".format(tensor_img.shape))
        # tensor_img = tensor_img.permute()
        tensor=tensor_img.resize_(1,3,224,224) # 变成 B x C x W x H(H x W) 的样子, 因为pytorch 只能处理带batch的数据
        tensor=tensor.cuda()#将数据发送到GPU，数据和模型在同一个设备上运行

        return tensor



    def main_proc(self):

        my_imagenet = MyImageNet()

        self.model = self.get_vgg_model()
        tensor_input = self.get_data()

        logits = self.model(tensor_input)
        probs_gpu = torch.nn.Softmax(dim=1)(logits)
        probs_cpu = probs_gpu.data.cpu().numpy()
        logger.info("sum of probs_cpu = {}".format(probs_cpu.sum()))
        logger.info("shape of probs_cpu = {}".format(probs_cpu.shape))

        topk = 10
        top_idxs = np.argsort(probs_cpu[0])[-1*topk:][::-1] # 从大到小排序
        # max_idx = np.argmax(probs_cpu[0])


        for id, idx in enumerate(top_idxs):
            str_label = my_imagenet.idx_to_label_dict[idx]
            prob = probs_cpu[0][idx]
            logger.info("top-{}: max_idx = {}, str_label = {}, prob = {}".format(id+1, idx, str_label, prob))


def main(argv):



    try:
        # main code
        logging_base_setting1()

        # str_src_img_path = "/data/dataset/dogs-vs-cats/train/cat.1144.jpg"
        # str_src_img_path = "/data/dataset/dogs-vs-cats/train/cat.1435.jpg"
        # str_src_img_path = "/data/dataset/dogs-vs-cats/train/dog.2456.jpg"
        # str_src_img_path = "/data/dataset/dogs-vs-cats/train/dog.12440.jpg"
        str_src_img_path = "/data/dataset/dogs-vs-cats/train/dog.12188.jpg"
        str_src_img_path = "/data/dataset/dogs-vs-cats/train/dog.12395.jpg"
        str_src_img_path = "/data/dataset/dogs-vs-cats/train/dog.11304.jpg"
        str_src_img_path = "/data/Dropbox/code/pt_base/data/tennisball1.jpg"   # good pic top3  resnet 152
        str_src_img_path = "/data/Dropbox/code/pt_base/data/dog1.jpg"   #
        str_src_img_path = "/data/Dropbox/code/pt_base/data/woman1.jpg"   # ?????   预处理的流程应该有问题!!!!!!
        official_model = OfficialModel(str_src_img_path=str_src_img_path)
        official_model.main_proc()


        logger.info("main finish")
    except Exception as exc:
        logger.exception("Unexpected exception! %s", exc)
    else:
        pass


if __name__ == '__main__':
    main(sys.argv)
