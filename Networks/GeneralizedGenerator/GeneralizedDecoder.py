# from Networks.NetworkClass import NetworkBase
import sys
sys.path.append('../')
sys.path.append('../../')


import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# import tensorflow.compat.v1 as tf # type: ignore
# tf.disable_v2_behavior()


import sys
sys.path.append('..')
cnnDim=64
vitDim=96
from Utilities.Blocks import patchSize

import numpy as np
import math
import torch
import torch.nn as nn
# from Utilities.ops import lrelu, relu, batch_norm, layer_norm, instance_norm, adaptive_instance_norm, resblock, desblock
# from Utilities.ops import conv2d, deconv2d, fc, dilated_conv2d, dilated_conv_resblock, normal_conv_resblock
# from Utilities.utils import PrintNetworkVars
from Utilities.Blocks import DecodingBasicBlock as BasicBlock
from Utilities.Blocks import DecodingBottleneckBlock as BottleneckBlock
from Utilities.Blocks import DecodingVisionTransformerBlock as VisionTransformerBlock

from prettytable import PrettyTable
from Utilities.utils import SplitName

BlockDict={'Cv': BasicBlock,
           'Cbb': BasicBlock,
           'Cbn': BottleneckBlock,
           'Vit': VisionTransformerBlock}

def FindKeys(dict, val): return list(value for key, value in dict.items() if key in val)
print_separater="#########################################################"


class GeneralizedDecoder(nn.Module):
    def __init__(self,  config, mixerFeatureShape):
        super(GeneralizedDecoder, self).__init__()
        self.config=config
        self.mixerFeatureShape=mixerFeatureShape
        self.decodingArchitectureList= SplitName(self.config.generator.decoder)[1:]
        self.encodingArchitectureList= SplitName(self.config.generator.encoder)[1:]
        downVitDim = (self.config.datasetConfig.imgWidth // patchSize )**2
        
        Block1 = FindKeys(BlockDict, self.decodingArchitectureList[0])[0]
        patchDim = -1
        outVitDim= -1
        mapC = cnnDim*16
        if 'Vit' in self.decodingArchitectureList[0]:
                outVitDim = int(mixerFeatureShape[3][1][1:-1].split(',')[-1])
                patchDim=outVitDim//2
                if 'C' in self.encodingArchitectureList[-1]:
                        mapC = cnnDim*16
                elif 'Vit' in self.encodingArchitectureList[-1]:
                        mapC = cnnDim*16  
                        # mapC = int(self.mixerFeatureShape[-1][0][1:-1].split(',')[1])
        elif 'C' in self.decodingArchitectureList[0]:
                if 'Vit' in self.encodingArchitectureList[-1]:
                        mapC = int(self.mixerFeatureShape[-1][0][1:-1].split(',')[1])
                if 'C' in self.encodingArchitectureList[-1]:
                        mapC = cnnDim*16
        self.decodingBlock1 = Block1(inDims={'HW':self.config.datasetConfig.imgWidth//16, 
                                              'MapC': mapC,
                                              'VitC': vitDim*32,
                                              'VitDim': downVitDim//256,
                                              'PatchDim': patchDim},
                                     outDims={'HW':self.config.datasetConfig.imgWidth//8, 
                                                             'MapC': cnnDim*8,
                                                             'VitC': vitDim*8,
                                                             'VitDim': outVitDim},
                                     config={'option': self.decodingArchitectureList[0],
                                             'upsample': True,
                                             'skip': None,
                                             'lastLayer': False})
        

        Block2 = FindKeys(BlockDict, self.decodingArchitectureList[1])[0]
        thisFeatureCNNShape = int(self.mixerFeatureShape[-2][0][1:-1].split(',')[1])
        thisFeatureVitShape = int(self.mixerFeatureShape[-2][1][1:-1].split(',')[-1])
        patchDim = -1
        outVitDim= -1
        # mapC=cnnDim*8
        if 'Vit' in self.decodingArchitectureList[1]:
                outVitDim = int(mixerFeatureShape[2][1][1:-1].split(',')[-1])
                if 'C' in self.decodingArchitectureList[0]:
                        outVitDim+=cnnDim*128
                        mapC=cnnDim*8
                elif 'Vit' in self.decodingArchitectureList[0]:
                        outVitDim+=vitDim*8
                        mapC=vitDim//2
                patchDim=outVitDim//2
        elif 'C' in self.decodingArchitectureList[1]:
                if 'C' in self.decodingArchitectureList[0]:
                        mapC=cnnDim*8   
                elif 'Vit' in self.decodingArchitectureList[0]:
                       mapC=vitDim//2  
                
        self.decodingBlock2 = Block2(inDims={'HW':self.config.datasetConfig.imgWidth//8, 
                                              'MapC': mapC,
                                              'VitC': vitDim*16,
                                              'VitDim': downVitDim//64,
                                              'PatchDim': patchDim},
                                     outDims={'HW':self.config.datasetConfig.imgWidth//4, 
                                                             'MapC': cnnDim*4,
                                                             'VitC': vitDim*4,
                                                             'VitDim': outVitDim},
                                     config={'option': self.decodingArchitectureList[1],
                                             'upsample': True,
                                             'skip': [thisFeatureCNNShape, thisFeatureVitShape],
                                             'lastLayer': False})
        
        
        
        Block3 = FindKeys(BlockDict, self.decodingArchitectureList[2])[0]
        thisFeatureCNNShape = int(self.mixerFeatureShape[-3][0][1:-1].split(',')[1])
        thisFeatureVitShape = int(self.mixerFeatureShape[-3][1][1:-1].split(',')[-1])
        patchDim = -1
        outVitDim= -1
        if 'Vit' in self.decodingArchitectureList[2]:
                outVitDim = int(mixerFeatureShape[1][1][1:-1].split(',')[-1])
                if 'C' in self.decodingArchitectureList[1]:
                        outVitDim+=cnnDim*64
                        mapC=cnnDim*4#
                elif 'Vit' in self.decodingArchitectureList[1]:
                        outVitDim+=vitDim*4
                        mapC=vitDim//4#
                patchDim=outVitDim//2
        elif 'C' in self.decodingArchitectureList[2]:
                if 'C' in self.decodingArchitectureList[1]:
                        mapC=cnnDim*4  
                elif 'Vit' in self.decodingArchitectureList[1]:
                       mapC=vitDim//4  
                       
                       
        self.decodingBlock3 = Block3(inDims={'HW':self.config.datasetConfig.imgWidth//4, 
                                              'MapC': mapC,
                                              'VitC': vitDim*8,
                                              'VitDim': downVitDim//4,
                                              'PatchDim': patchDim},
                                     outDims={'HW':self.config.datasetConfig.imgWidth//2, 
                                                             'MapC': cnnDim*2,
                                                             'VitC': vitDim*2,
                                                             'VitDim': outVitDim},
                                     config={'option': self.decodingArchitectureList[2],
                                             'upsample': True,
                                             'skip': [thisFeatureCNNShape, thisFeatureVitShape],
                                             'lastLayer': False})
        
        
        Block4 = FindKeys(BlockDict, self.decodingArchitectureList[3])[0]
        patchDim = -1
        outVitDim= -1
        if 'Vit' in self.decodingArchitectureList[3]:
                outVitDim = int(mixerFeatureShape[0][1][1:-1].split(',')[-1])
                if 'C' in self.decodingArchitectureList[2]:
                        outVitDim+=cnnDim*32
                        mapC=cnnDim*2#
                elif 'Vit' in self.decodingArchitectureList[2]:
                        outVitDim+=vitDim*2
                        mapC=vitDim//8#
                patchDim=outVitDim//2
        elif 'C' in self.decodingArchitectureList[3]:
                if 'C' in self.decodingArchitectureList[2]:
                        mapC=cnnDim*2  
                elif 'Vit' in self.decodingArchitectureList[2]:
                       mapC=vitDim//8  
                        
        #patchDim = cnnDim*16
        thisFeatureCNNShape = int(self.mixerFeatureShape[-4][0][1:-1].split(',')[1])
        thisFeatureVitShape = int(self.mixerFeatureShape[-4][1][1:-1].split(',')[-1])
        self.decodingBlock4 = Block4(inDims={'HW':self.config.datasetConfig.imgWidth//2, 
                                              'MapC': mapC,
                                              'VitC': vitDim*2,
                                              'VitDim': downVitDim*2,
                                              'PatchDim': patchDim},
                                     outDims={'HW':self.config.datasetConfig.imgWidth, 
                                                             'MapC': cnnDim,
                                                             'VitC': vitDim,
                                                             'VitDim': outVitDim},
                                     config={'option': self.decodingArchitectureList[3],
                                             'upsample': True,
                                             'skip': [thisFeatureCNNShape, thisFeatureVitShape],
                                             'lastLayer': False})
        
        
        
        if 'Vit' in self.decodingArchitectureList[3]:
                mapC = vitDim//16
        elif 'C' in self.decodingArchitectureList[3]:
                mapC= cnnDim
        self.generated = BasicBlock(inDims={'HW':self.config.datasetConfig.imgWidth, 
                                              'MapC': mapC,
                                              'VitC': -1,
                                              'VitDim': -1,
                                              'PatchDim': -1},
                                     outDims={'HW':self.config.datasetConfig.imgWidth, 
                                                             'MapC': self.config.datasetConfig.channels,
                                                             'VitC': -1,
                                                             'VitDim': -1},
                                     config={'option': self.decodingArchitectureList[3],
                                             'upsample': False,
                                             'skip': None,
                                             'lastLayer': True})
        
        
    def forward(self,encodedLayers):
        result1=self.decodingBlock1(encodedLayers[-1])
        result2=self.decodingBlock2(result1, encodedLayers[-2])
        result3=self.decodingBlock3(result2, encodedLayers[-3])
        result4=self.decodingBlock4(result3, encodedLayers[-4])
        finalGenerated = self.generated(result4).cnn
        
        return [finalGenerated,result4,result3,result2,result1]

    

        