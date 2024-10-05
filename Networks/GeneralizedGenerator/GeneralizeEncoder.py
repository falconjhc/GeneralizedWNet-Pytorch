# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.append('../')
sys.path.append('../../')


import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


import numpy as np 
import math

# from Utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock, maxPool
# from Utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock

from Utilities.Blocks import EncodingBasicBlock as BasicBlock, EncodingBottleneckBlock as BottleneckBlock, PatchExtractor
from Utilities.utils import SplitName
from Utilities.Blocks import BlockFeature
from Utilities.Blocks import EncodingVisionTransformerBlock as VisionTransformerBlock


eps = 1e-9
cnnDim = 64
vitDim = 96
from Utilities.Blocks import patchSize
BlockDict={'Cv': BasicBlock,
           'Cbb': BasicBlock,
           'Cbn': BottleneckBlock,
           'Vit': VisionTransformerBlock}
def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)
numHeads=[3,6,12,24]
depths=[2,2,2,2]
# numHeads=[-1, -1,-1,8,-1]
# depths=[-1, -1,-1,12,-1]

print_separater="#########################################################"


class GeneralizedEncoder(nn.Module):
    def __init__(self,  config, loadedCategoryLength=80 , mark='NA'):
        super(GeneralizedEncoder, self).__init__()
        self.config=config
        self.architectureList= SplitName(self.config.generator.encoder)[1:]
        
        # Block 0
        if mark=='Content':
            mapC0=self.config.datasetConfig.inputContentNum*self.config.datasetConfig.channels
        elif mark=='Style':
            mapC0=self.config.datasetConfig.channels
        self.patchExtractor = PatchExtractor(featureDim=self.config.datasetConfig.channels)
        self.encodingBlock0 = BottleneckBlock(inDims={'HW':self.config.datasetConfig.imgWidth, 
                                                      'MapC': mapC0,
                                                      'VitC': -1, 'VitDim': -1}, 
                                              outDims= {'HW':self.config.datasetConfig.imgWidth, 
                                                        'MapC': cnnDim//2,
                                                        'VitC': -1, 'VitDim': -1},
                                              config={'downsample': False})
        
       
        
        # Block 1
        patchSize=4
        downVitDim = (self.config.datasetConfig.imgWidth // patchSize )**2
        Block1 = FindKeys(BlockDict, self.architectureList[0])[0]
        self.encodingBlock1 = Block1(inDims={'HW':self.config.datasetConfig.imgWidth, 
                                                             'MapC': cnnDim//2,
                                                             'VitC': -1,
                                                             'VitDim': -1,
                                                             'PatchDim': 512}, 
                                     outDims={'HW':self.config.datasetConfig.imgWidth//2, 
                                                             'MapC': cnnDim,
                                                             'VitC': vitDim,
                                                             'VitDim': downVitDim//4}, 
                                     config={'option': self.architectureList[0],
                                             'downsample': True})
        
        # Block 2
        Block2 = FindKeys(BlockDict, self.architectureList[1])[0]
        if 'Vit' in self.architectureList[0]:
            patchDim = vitDim*2
            mapC=vitDim//8
        else:
            patchDim = cnnDim*16
            mapC = cnnDim
        self.encodingBlock2 = Block2(inDims={'HW':self.config.datasetConfig.imgWidth//2, 
                                                             'MapC': mapC,
                                                             'VitC': vitDim,
                                                             'VitDim': downVitDim//4,
                                                             'PatchDim': patchDim}, 
                                     outDims={'HW': self.config.datasetConfig.imgWidth//4, 
                                                             'MapC': cnnDim*2,
                                                             'VitC': vitDim*2,
                                                             'VitDim': downVitDim//16}, 
                                     config={'option': self.architectureList[1],
                                             'downsample': True})
        
        # Block 3
        Block3 = FindKeys(BlockDict, self.architectureList[2])[0]
        if 'Vit' in self.architectureList[1]:
            patchDim = vitDim*4
            mapC=vitDim//4
        else:
            patchDim = cnnDim*32
            mapC = cnnDim*2
        self.encodingBlock3 = Block3(inDims={'HW': self.config.datasetConfig.imgWidth//4, 
                                                             'MapC': mapC,
                                                             'VitC': vitDim*2,
                                                             'VitDim': downVitDim//16,
                                                             'PatchDim': patchDim}, 
                                     outDims={'HW':self.config.datasetConfig.imgWidth//8, 
                                                             'MapC': cnnDim*4,
                                                             'VitC': vitDim*4,
                                                             'VitDim': downVitDim//64}, 
                                     config={'option': self.architectureList[2],
                                             'downsample': True})
        
        # Block 4
        Block4 = FindKeys(BlockDict, self.architectureList[3])[0]
        if 'Vit' in self.architectureList[2]:
            patchDim = vitDim*8
            mapC=vitDim//2
        else:
            patchDim = cnnDim*64
            mapC=cnnDim*4
        self.encodingBlock4 = Block4(inDims={'HW':self.config.datasetConfig.imgWidth//8, 
                                                             'MapC': mapC,
                                                             'VitC': vitDim*4,
                                                             'VitDim': downVitDim//64,
                                                             'PatchDim': patchDim},
                                     outDims={'HW':self.config.datasetConfig.imgWidth//16, 
                                              'MapC': cnnDim*8,
                                              'VitC': vitDim*8,
                                              'VitDim': downVitDim//256}, 
                                     config={'option': self.architectureList[3],
                                             'downsample': True})
        
        
        
        #Block Category
        if 'Vit' in self.architectureList[3]:
            patchDim = vitDim*16
            mapC=vitDim
        else:
            patchDim = cnnDim*128
            mapC=cnnDim*8
        if mark=='Content':
            self.category = nn.Sequential(nn.Conv2d(in_channels=mapC, out_channels=256, kernel_size=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(True),
                                        nn.Flatten(1,3),
                                        nn.Linear(4096, loadedCategoryLength))
        elif mark=='Style':
            self.category = nn.Sequential(nn.Conv2d(in_channels=mapC, out_channels=128, kernel_size=1),
                                          nn.BatchNorm2d(128),
                                          nn.ReLU(True),
                                          nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1),
                                          nn.BatchNorm2d(32),
                                          nn.ReLU(True),
                                          nn.Flatten(1,3),
                                          nn.Linear(512, loadedCategoryLength)
                                          )
        
        
        
        
    def forward(self, xcnn):
        # STEM
        xvit = self.patchExtractor(xcnn)
        x = BlockFeature(cnn=xcnn,vit=xvit)
        x0 = self.encodingBlock0(x)
        
        # Block 1
        x1 = self.encodingBlock1(x0)
        
        # Block 2
        x2 = self.encodingBlock2(x1)
        
        # Block 3
        x3 = self.encodingBlock3(x2)
        
        # Block 4
        x4 = self.encodingBlock4(x3)
        
        
        # categories
        categoryOut = self.category(x4.cnn)
        
        return categoryOut, [x0,x1,x2,x3,x4]
        

