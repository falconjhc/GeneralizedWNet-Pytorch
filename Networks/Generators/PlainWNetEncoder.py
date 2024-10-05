import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


sys.path.append('./')
from Networks.utils.ops import Convblock
from Pipelines.Dataset import CASIA_Dataset
from config import CONFIG as cfg

class Encoder(nn.Module):
    def __init__(self, normalization, loadedCategoryLength=80 ,input_channels = 1, generator_dim = 64):
        super(Encoder, self).__init__()
        # self.config = config
        self.normalization = normalization
        #1*64*64
        self.conv2d_one = nn.Conv2d(input_channels, generator_dim, kernel_size=5, stride=2, padding=2) #64*32*32
        self.encoderblock_one = Convblock(generator_dim, generator_dim*2, normalization=normalization) #128*16*16
        self.encoderblock_two = Convblock(generator_dim*2, generator_dim*4, normalization=normalization) #256*8*8
        self.encoderblock_three = Convblock(generator_dim*4, generator_dim*8, normalization=normalization) #512*2*2
        self.encoderblock_four = Convblock(generator_dim*8, generator_dim*8, normalization=normalization) #512*2*2
        self.encoderblock_five = Convblock(generator_dim*8, generator_dim*8, normalization=normalization) #512*1*1
        self.fc = nn.Linear(generator_dim*8, loadedCategoryLength)


    def forward(self, input):
        x1 = self.conv2d_one(input)
        x2 = self.encoderblock_one(x1)
        x3 = self.encoderblock_two(x2)
        x4 = self.encoderblock_three(x3)
        x5 = self.encoderblock_four(x4)
        x6 = self.encoderblock_five(x5)
        x7 = x6.view(x6.size(0), -1)
        output = nn.functional.relu(x7)
        category = self.fc(output)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        # print(x4.shape)
        # print(x5.shape)
        # print(x6.shape)
        # print(category.shape)
        res = [x1,x2,x3,x4,x5,x6]
        return category,res
    
if __name__ == '__main__':
    # x = torch.randn(16,1,64,64)
    # input_channels = 1
    # generator_dim  = 64
    # conv_layer = Encoder(input_channels=input_channels,generator_dim=generator_dim ,normalization='batchnorm')
    # x = conv_layer(x)
    
    cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
    cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml'

    input_channels = 64
    generator_dim  = 64
    batchsize = 8
    # 创建CASIA数据集实例
    casia_dataset = CASIA_Dataset(cfg)
    # 创建DataLoader
    casia_loader = DataLoader(casia_dataset, batch_size=batchsize, shuffle=False)
    conv_layer = Encoder(input_channels=input_channels,generator_dim=generator_dim ,normalization='batchnorm')

    # 读入第一个样本
    for contents, styles, GT_style,_,_ in casia_loader:
        # contents = contents
        # reshape_contents = contents.reshape(512, 1, 64, 64)
        categories,res = conv_layer(contents)

        print(styles.shape)

        print(GT_style.shape)