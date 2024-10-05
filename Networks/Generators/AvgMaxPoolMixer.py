import os
import sys
# sys.path.append('./Networks')
# sys.path.append('./Networks')

import torch
import torch.nn as nn
import numpy as np
from config import CONFIG as cfg

from Networks.utils.ops import ResidualConvBlock

class WNetMixer(nn.Module):
    def __init__(self, normalization,generator_dim = 64):
        super(WNetMixer, self).__init__()
        self.M = 5
        self.ResidualConvBlock_one = ResidualConvBlock(self.M-4,generator_dim, generator_dim, normalization=normalization) #64
        self.ResidualConvBlock_two = ResidualConvBlock(self.M-2,generator_dim*2, generator_dim*2, normalization=normalization) #128
        self.ResidualConvBlock_three = ResidualConvBlock(self.M,generator_dim*4, generator_dim*4, normalization=normalization) #256


    def forward(self, inputs1,inputs2):
        # max_input1 = [torch.max(x, dim=1)[0] for x in inputs1]
        # max_input2 = [torch.max(x, dim=1)[0] for x in inputs2]
        input1_x1,input1_x2,input1_x3,input1_x4,input1_x5,input1_x6 = inputs1
        input2_x1,input2_x2,input2_x3,input2_x4,input2_x5,input2_x6 = inputs2
        mixer1 = self.ResidualConvBlock_one(input1_x1)
        mixer2 = self.ResidualConvBlock_two(input1_x2)
        mixer3 = torch.concat([self.ResidualConvBlock_three(input1_x3),self.ResidualConvBlock_three(input2_x3)],dim=1)
        mixer4 = torch.concat([input1_x4,input2_x4],dim=1)
        mixer5 = torch.concat([input1_x5,input2_x5],dim=1)
        mixer6 = torch.concat([input1_x6,input2_x6],dim=1)
        res = [mixer1,mixer2,mixer3,mixer4,mixer5,mixer6]

        return res
    
    def reshape_tensor(self,input_tensor,is_train):
        if is_train:
            batchsize = self.batchsize
        else:
            batchsize = self.val_batchsize

        if len(input_tensor.shape) == 4:
            return input_tensor.reshape(batchsize,input_tensor.shape[0]//batchsize,input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3])
        elif len(input_tensor.shape) == 3:
            return input_tensor.reshape(batchsize,input_tensor.shape[0]//batchsize,input_tensor.shape[1],input_tensor.shape[2])
        elif len(input_tensor.shape) == 2:
            return input_tensor.reshape(batchsize,input_tensor.shape[0]//batchsize,input_tensor.shape[1])
    
if __name__ == '__main__':
    inputs1 = [torch.randn(16,64,32,32),
            torch.randn(16,128,16,16),
            torch.randn(16,256,8,8),
            torch.randn(16,512,4,4),
            torch.randn(16,512,2,2),
            torch.randn(16,512,1,1)]
    inputs2 = [torch.randn(16,64,32,32),
            torch.randn(16,128,16,16),
            torch.randn(16,256,8,8),
            torch.randn(16,512,4,4),
            torch.randn(16,512,2,2),
            torch.randn(16,512,1,1)]
    mixer = WNetMixer(normalization='batchnorm')
    res = mixer(inputs1,inputs2)
    for x in res:
        print(x.shape)


