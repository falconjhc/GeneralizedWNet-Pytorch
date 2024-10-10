# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function


import sys
sys.path.append('../')
sys.path.append('../../')


import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import torch
import torch.nn as nn

torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()
from Utilities.utils import SplitName

import numpy as np 
import math

from Networks.GeneralizedGenerator.NetworkClass import NetworkIO

from Networks.GeneralizedGenerator.GeneralizeEncoder import GeneralizedEncoder as Encoder
from Networks.GeneralizedGenerator.GeneralizedMixer import WNetMixer as Mixer
from Networks.GeneralizedGenerator.GeneralizedDecoder import GeneralizedDecoder as Decoder
# from Networks.GeneralizedGenerator.GeneralizedDecoder import GeneralizedDecoder as Decoder
# from Networks.GeneralizedGenerator.GeneralizedMixer import WNetMixer as Mixer

# import os 
from Networks.GeneralizedGenerator.ModuleBase import ModuleBase
from Networks.utils.ops import set_random
from prettytable import PrettyTable


eps = 1e-9
generator_dim = 64
print_separater="#########################################################"

 
class WNetGeneratorBase(nn.Module):
    def __init__(self, config):
        super(WNetGeneratorBase, self).__init__()    
        
        self.config=config
        self.is_train=True
        set_random()
        self.encodingArchitectureList=SplitName(self.config.generator.encoder)[1:]
        self.decodingArchitectureList=SplitName(self.config.generator.decoder)[1:]
        self.encodingBlockNum=len(self.encodingArchitectureList)+1
        
        self.contentEncoder=Encoder(config=config,
                                     loadedCategoryLength=len(config.datasetConfig.loadedLabel0Vec),
                                     mark='Content')
        self.contentEncoder.train()
        self.contentEncoder.cuda()
        
        self.styleEncoder=Encoder(config=config,
                                  loadedCategoryLength=len(config.datasetConfig.loadedLabel1Vec),
                                  mark='Style')
        self.styleEncoder.train()
        self.styleEncoder.cuda()

        contentFeatures,styleFeatureList,contentCategory, styleCategoryList, encodingFeatureShape = self.TestEncoders()
        
        
        self.mixer = Mixer(encodingFeatureShape=encodingFeatureShape, config=self.config)
        self.mixer.train()
        self.mixer.cuda()
        
        fusedFinal, mixerFeatureShape=self.TestMixer(styleFeatures=styleFeatureList, contentFeatures=contentFeatures)
        
        
        self.decoder = Decoder(config=config, mixerFeatureShape=mixerFeatureShape)
        self.decoder.train()
        self.decoder.cuda()
        
        
        
        finalGenerated = self.TestDecoder(fusedFinal)
        print(print_separater)
        print("Architecture Construction Completed")
        print(print_separater)
        



        
        
    def TestEncoders(self):
        
        testContent = torch.randn(self.config.trainParams.batchSize, self.config.datasetConfig.inputContentNum, 64, 64).to('cuda')  
        testStyle = torch.randn(self.config.trainParams.batchSize, self.config.datasetConfig.inputStyleNum, 64, 64).to('cuda')  
        
        
        contentCategory, contentFeatures = self.contentEncoder(testContent)
        print(print_separater)
        print("Content Encoder Architecture")
        table = PrettyTable(['Layer', 'CNN','VIT'])
        table.add_row(['0-STEM-Bottleneck']+ contentFeatures[0].ProcessOutputToList() )
        table.add_row(['1-'+self.encodingArchitectureList[0]]+ contentFeatures[1].ProcessOutputToList() )
        table.add_row(['2-'+self.encodingArchitectureList[1]]+ contentFeatures[2].ProcessOutputToList() )
        table.add_row(['3-'+self.encodingArchitectureList[2]]+ contentFeatures[3].ProcessOutputToList() )
        table.add_row(['4-'+self.encodingArchitectureList[3]]+ contentFeatures[4].ProcessOutputToList() )
        print(table)     
        
        encodingFeatureShape = [contentFeatures[0].ProcessOutputToList(),
                                contentFeatures[1].ProcessOutputToList(),
                                contentFeatures[2].ProcessOutputToList(),
                                contentFeatures[3].ProcessOutputToList(),
                                contentFeatures[4].ProcessOutputToList()]
        
        
        styleCategoryList=[]
        styleFeatureList=[]
        for ii in range(self.encodingBlockNum):
            styleFeatureList.append([])
        for ii in range(self.config.datasetConfig.inputStyleNum):
            thisStyleInput = torch.unsqueeze(testStyle[:,ii,:,:],1)
            thisStyleCategory, thisStyleFeatures = self.styleEncoder(thisStyleInput)
            styleCategoryList.append(thisStyleCategory)
            
            
            for jj in range(len(thisStyleFeatures)):
                styleFeatureList[jj].append(thisStyleFeatures[jj])
        print(print_separater)
        print("Style Encoder Architecture")
        table = PrettyTable(['Layer', 'CNN','VIT'])
        table.add_row(['0-STEM-Bottleneck']+ styleFeatureList[0][0].ProcessOutputToList() )
        table.add_row(['1-'+self.encodingArchitectureList[0]]+ styleFeatureList[1][0].ProcessOutputToList())
        table.add_row(['2-'+self.encodingArchitectureList[1]]+ styleFeatureList[2][0].ProcessOutputToList())
        table.add_row(['3-'+self.encodingArchitectureList[2]]+ styleFeatureList[3][0].ProcessOutputToList())
        table.add_row(['4-'+self.encodingArchitectureList[3]]+ styleFeatureList[4][0].ProcessOutputToList())
        print(table)     
        print(print_separater)
        
        
        return contentFeatures,styleFeatureList,contentCategory, styleCategoryList, encodingFeatureShape
        
        
        
    def TestMixer(self, styleFeatures, contentFeatures):
        fusedFinal=self.mixer(styleFeatures=styleFeatures, contentFeatures=contentFeatures)
        print(print_separater)
        print("Mixer Architecture")
        table = PrettyTable(['Layer', 'CNN','VIT'])
        table.add_row(['1-'+self.encodingArchitectureList[0]]+ fusedFinal[0].ProcessOutputToList() )
        table.add_row(['2-'+self.encodingArchitectureList[1]]+ fusedFinal[1].ProcessOutputToList() )
        table.add_row(['3-'+self.encodingArchitectureList[2]]+ fusedFinal[2].ProcessOutputToList() )
        table.add_row(['4-'+self.encodingArchitectureList[3]]+ fusedFinal[3].ProcessOutputToList() )
        print(table)     
        mixerFeatureShape = [fusedFinal[0].ProcessOutputToList(),
                                fusedFinal[1].ProcessOutputToList(),
                                fusedFinal[2].ProcessOutputToList(),
                                fusedFinal[3].ProcessOutputToList()]
        
        
        return fusedFinal, mixerFeatureShape
        
    def TestDecoder(self, encoded):
        decoderList=self.decoder(encoded)
        generated = decoderList[0]
        
        print(print_separater)
        print("Decoder Architecture")
        table = PrettyTable(['CNN','VIT','Layer'])
        table.add_row([str(generated.shape)[11:-1]]+['NA']+ ['5-Generated-BasicBlock'])
        table.add_row(decoderList[1].ProcessOutputToList() +['4-'+self.decodingArchitectureList[-1]])
        table.add_row(decoderList[2].ProcessOutputToList() +['3-'+self.decodingArchitectureList[-2]])
        table.add_row(decoderList[3].ProcessOutputToList() +['2-'+self.decodingArchitectureList[-2]])
        table.add_row(decoderList[4].ProcessOutputToList() +['1-'+self.decodingArchitectureList[-3]])
        print(table)     
        
        
        return generated
        
    def forward(self,content_inputs,style_inputs,GT, is_train=True):
        contentCategory_onReal, contentFeatures_onReal = self.contentEncoder(content_inputs)        
        styleCategoryFull_onReal=[]
        styleFeatureList_onReal=[]
        for ii in range(self.encodingBlockNum):
            styleFeatureList_onReal.append([])
        
        style_inputs = style_inputs.reshape((self.config.trainParams.batchSize, self.config.datasetConfig.inputStyleNum, 
                                             self.config.datasetConfig.imgWidth,self.config.datasetConfig.imgWidth))
        for ii in range(self.config.datasetConfig.inputStyleNum):
            this_style_category_onReal, this_style_outputs_onReal = self.styleEncoder(torch.unsqueeze(style_inputs[:,ii,:,:], dim=1))
            if ii ==0:
                styleCategoryFull_onReal = this_style_category_onReal
            else:
                styleCategoryFull_onReal = torch.concat((styleCategoryFull_onReal,this_style_category_onReal), dim=0)
            for jj in range(len(this_style_outputs_onReal)):
                styleFeatureList_onReal[jj].append(this_style_outputs_onReal[jj])
        
        
        
        
        reshaped_style_list_onReal=[]
        for ii in range(len(styleFeatureList_onReal)):
            for jj in range(len(styleFeatureList_onReal[ii])):
                if jj ==0:
                    thisFeature = torch.unsqueeze(styleFeatureList_onReal[ii][jj].cnn,1)
                else:
                    thisFeature = torch.concat((thisFeature,torch.unsqueeze(styleFeatureList_onReal[ii][jj].cnn,1)),1)
            reshaped_style_list_onReal.append(thisFeature)
        enc_content_list_onReal= [ii.cnn for ii in contentFeatures_onReal]

        
        mix_output = self.mixer(styleFeatures=styleFeatureList_onReal, contentFeatures=contentFeatures_onReal)
        decode_output_list=self.decoder(mix_output)
        for ii in range(len(decode_output_list)):
            if ii ==0:
                continue
            else:
               decode_output_list[ii]=decode_output_list[ii].cnn 
        decode_output_list.reverse()
        generated = decode_output_list[-1]
        
        
        # encoded the groundtruth
        GT_content_category,GT_content_outputs = self.contentEncoder(GT.repeat(1, GT.shape[1]*self.config.datasetConfig.inputContentNum, 1, 1))
        GT_style_category,GT_style_outputs = self.styleEncoder(GT)
        
        
        # encode the generated
        contentCategoryOnGenerated, contentFeaturesOnGenerated = self.contentEncoder(generated.repeat((1,self.config.datasetConfig.inputContentNum,1,1)))
        enc_content_onGenerated_list= [ii.cnn for ii in contentFeaturesOnGenerated]
        styleCategoryOnGenerated, styleFeaturesOnGenerated = self.styleEncoder(generated)
        enc_style_onGenerated_list= [ii.cnn for ii in styleFeaturesOnGenerated]
        contentFeaturesOnGenerated=enc_content_onGenerated_list
        styleFeaturesOnGenerated = enc_style_onGenerated_list
        
        
        
        # process
        max_content_category_onReal = torch.max(self.reshape_tensor(contentCategory_onReal, is_train),dim=1)[0] 
        max_style_category_onReal = torch.max(self.reshape_tensor(styleCategoryFull_onReal, is_train),dim=1)[0]
        max_content_category_onGenerated = torch.max(self.reshape_tensor(contentCategoryOnGenerated, is_train),dim=1)[0]
        max_style_category_onGenerated = torch.max(self.reshape_tensor(styleCategoryOnGenerated, is_train),dim=1)[0]
        
        encodedContentFeatures={}
        encodedStyleFeatures={}
        encodedContentCategory={}
        encodedStyleCategory={}
        
        encodedContentFeatures.update({'real': enc_content_list_onReal[-1]})
        encodedContentFeatures.update({'fake': contentFeaturesOnGenerated[-1]})
        encodedContentFeatures.update({'groundtruth': GT_content_outputs[-1].cnn})
        
        encodedStyleFeatures.update({'real': reshaped_style_list_onReal[-1]})
        encodedStyleFeatures.update({'fake': styleFeaturesOnGenerated[-1]})
        encodedStyleFeatures.update({'groundtruth': GT_style_outputs[-1].cnn})
        
        encodedContentCategory.update({'real': max_content_category_onReal})
        encodedContentCategory.update({'fake': max_content_category_onGenerated})
        encodedContentCategory.update({'groundtruth': GT_content_category})
        
        encodedStyleCategory.update({'real': max_style_category_onReal})
        encodedStyleCategory.update({'fake': max_style_category_onGenerated})
        encodedStyleCategory.update({'groundtruth': GT_style_category})
        
        
        return encodedContentFeatures, encodedStyleFeatures, encodedContentCategory, encodedStyleCategory, generated

    
        
    def reshape_tensor(self,input_tensor,is_train):
        if len(input_tensor.shape) == 4:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3])
        elif len(input_tensor.shape) == 3:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1],input_tensor.shape[2])
        elif len(input_tensor.shape) == 2:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1])
    