import sys
import torch
torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()



import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

sys.path.append('./')
from Networks.Generators.AvgMaxPoolMixer import WNetMixer
from Networks.Generators.PlainWNetDecoder import Decoder
from Networks.Generators.PlainWNetEncoder import Encoder
from Networks.utils.ops import set_random
from Pipelines.Dataset import CASIA_Dataset
from config import CONFIG as cfg

cnnDim = 64
vitDim = 96

class WNetGenerator(nn.Module):
    def __init__(self, config):
        super(WNetGenerator, self).__init__()    
                  
        # config
        self.is_train = True
        self.config=config
        #self.batchsize = config.trainParams.batchSize
        # self.val_batchsize = config['val_batch_size']              
        # self.in_channels = config['in_channels']
        #self.generator_dim = config['generator_dim']
        # self.inputContentNum = self.config.datasetConfig.inputContentNum
        # self.inputStyleNum = config['inputStyleNum']
        set_random()
        self.content_encoder = Encoder(input_channels=self.config.datasetConfig.channels*self.config.datasetConfig.inputContentNum,
                                       loadedCategoryLength=len(config.datasetConfig.loadedLabel0Vec),
                                       generator_dim=cnnDim,normalization='batchnorm')
        self.content_encoder.train()
        self.content_encoder.cuda()
        
        self.style_encoder = Encoder(input_channels=self.config.datasetConfig.channels,
                                     loadedCategoryLength=len(config.datasetConfig.loadedLabel1Vec),
                                     generator_dim=cnnDim,normalization='batchnorm')
        self.style_encoder.train()
        self.style_encoder.cuda()
        
        self.mixer = WNetMixer(generator_dim=cnnDim,normalization='batchnorm')
        self.mixer.train()
        self.mixer.cuda()

        self.decoder = Decoder(out_channels=self.config.datasetConfig.channels,generator_dim=cnnDim,normalization='batchnorm')
        self.decoder.train()
        self.decoder.cuda()
    
    def forward(self, content_inputs,style_inputs,GT, is_train=True):
        # encode
        content_category,content_outputs = self.content_encoder(content_inputs)
        reshaped_content_category = self.reshape_tensor(content_category, is_train)
        max_content_category = torch.max(reshaped_content_category,dim=1)[0]
        
        
        
        repeat_GT =  GT.repeat(1, GT.shape[1]*self.config.datasetConfig.inputContentNum, 1, 1)
        GT_content_category,GT_content_outputs = self.content_encoder(repeat_GT)
        GT_content_last = GT_content_outputs[-1]

        style_category,style_outputs = self.style_encoder(style_inputs)
        reshaped_style_category = self.reshape_tensor(style_category, is_train)
        max_style_category = torch.max(reshaped_style_category,dim=1)[0]
        GT_style_category,GT_style_outputs = self.style_encoder(GT)
        GT_style_last = GT_style_outputs[-1]

        GT_output = (GT_content_last,GT_style_last,GT_content_category,GT_style_category)

        enc_content_list,enc_style_list = [],[] 
        reshaped_style_list = []
        for content,style in zip(content_outputs,style_outputs):
            # reshaped_content_outputs = self.reshape_tensor(content,is_train)
            # max_content_output = torch.max(reshaped_content_outputs,dim=1)[0]
            reshaped_style_outputs = self.reshape_tensor(style,is_train)
            max_style_output = torch.max(reshaped_style_outputs,dim=1)[0]

            # 将64个相同content的encoder结果加入列表，用于mix
            enc_content_list.append(content)

            # 将5个相同style的encoder结果的最大值加入列表，用于mix
            enc_style_list.append(max_style_output)
            # 将5个相同style的encoder结果的每个值加入列表
            reshaped_style_list.append(reshaped_style_outputs)

        # mix
        mix_output = self.mixer(enc_content_list,enc_style_list)

        # decode
        decode_output_list = self.decoder(mix_output)
        
        
        
        generated = decode_output_list[-1]
        contentCategoryOnGenerated, contentFeaturesOnGenerated = self.content_encoder(generated.repeat((1,self.config.datasetConfig.inputContentNum,1,1)))
        reshaped_content_category_onGenerated = self.reshape_tensor(contentCategoryOnGenerated, is_train)
        max_content_category_onGenerated = torch.max(reshaped_content_category_onGenerated,dim=1)[0]
        # = [ii.cnn for ii in contentFeaturesOnGenerated]
        
        
        styleCategoryOnGenerated, styleFeaturesOnGenerated = self.style_encoder(generated)
        # enc_style_onGenerated_list= [ii.cnn for ii in styleFeaturesOnGenerated]
        # styleFeaturesOnGenerated=styleFeaturesOnGenerated.cnn
        reshaped_style_category_onGenerated = self.reshape_tensor(styleCategoryOnGenerated, is_train)
        max_style_category_onGenerated = torch.max(reshaped_style_category_onGenerated,dim=1)[0]
        
        
        return enc_content_list, max_content_category,contentFeaturesOnGenerated,max_content_category_onGenerated, \
                reshaped_style_list, max_style_category,styleFeaturesOnGenerated,max_style_category_onGenerated, \
                decode_output_list, GT_output #style的loss需要5个都算
    
    def reshape_tensor(self,input_tensor,is_train):
        # if is_train:
        #     batchsize = self.batchsize
        # else:
        #     batchsize = self.val_batchsize

        if len(input_tensor.shape) == 4:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1],input_tensor.shape[2],input_tensor.shape[3])
        elif len(input_tensor.shape) == 3:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1],input_tensor.shape[2])
        elif len(input_tensor.shape) == 2:
            return input_tensor.reshape(self.config.trainParams.batchSize,input_tensor.shape[0]//self.config.trainParams.batchSize,input_tensor.shape[1])
    
if __name__ == '__main__':
    cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
    cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml' 
    cfg['val_GT_style_yaml'] = 'cmy/test_list/val_GT_dir.yaml'
    cfg['val_reference_style_yaml'] = 'cmy/test_list/val_reference_style_dir.yaml' 
    cfg['batch_size'] = 8
    cfg['val_batch_size'] = 8

    batchsize = 8
     # 创建CASIA数据集实例
    casia_dataset = CASIA_Dataset(cfg)
    # 创建DataLoader
    casia_loader = DataLoader(casia_dataset, batch_size=batchsize, shuffle=False,drop_last=True)
    Wnet = WNetGenerator(cfg)
    # 读入第一个样本
    for contents, styles, GT_style in casia_loader:
        contents, styles, GT_style = contents.cuda(), styles.cuda(), GT_style.cuda() 
        # reshape_contents = contents.reshape(batchsize*64, 1, 64, 64)

        reshape_styles = styles.reshape(batchsize*5,1,64,64)

        enc_content_list,content_category, \
        reshaped_enc_style_list, style_category, \
        decode_output_list,GT_output = Wnet(contents,reshape_styles,GT_style)

        print(decode_output_list[-2].shape)
        
