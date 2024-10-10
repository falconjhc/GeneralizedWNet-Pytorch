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
        content_category_onReal,content_outputs_onReal = self.content_encoder(content_inputs)
        style_category_onReal,style_outputs_onReal = self.style_encoder(style_inputs)
        enc_content_list_onReal,enc_style_list_onReal = [],[] 
        reshaped_style_list_onReal = []
        for content,style in zip(content_outputs_onReal,style_outputs_onReal):
            reshaped_style_outputs = self.reshape_tensor(style,is_train)
            max_style_output = torch.max(reshaped_style_outputs,dim=1)[0]
            enc_content_list_onReal.append(content)
            enc_style_list_onReal.append(max_style_output)
            reshaped_style_list_onReal.append(reshaped_style_outputs)

        # mix
        mix_output = self.mixer(enc_content_list_onReal,enc_style_list_onReal)
        
        # decode
        decode_output_list = self.decoder(mix_output)
        generated = decode_output_list[-1]

        # encode the groundtruth
        GT_content_category,GT_content_outputs = self.content_encoder(GT.repeat(1, GT.shape[1]*self.config.datasetConfig.inputContentNum, 1, 1))
        GT_style_category,GT_style_outputs = self.style_encoder(GT)
        
        # encode the generated
        contentCategoryOnGenerated, contentFeaturesOnGenerated = self.content_encoder(generated.repeat((1,self.config.datasetConfig.inputContentNum,1,1)))
        styleCategoryOnGenerated, styleFeaturesOnGenerated = self.style_encoder(generated)
        
        

        # process
        max_content_category_onReal = torch.max(self.reshape_tensor(content_category_onReal, is_train),dim=1)[0]
        max_style_category_onReal = torch.max(self.reshape_tensor(style_category_onReal, is_train),dim=1)[0]
        max_content_category_onGenerated = torch.max(self.reshape_tensor(contentCategoryOnGenerated, is_train),dim=1)[0]
        max_style_category_onGenerated = torch.max(self.reshape_tensor(styleCategoryOnGenerated, is_train),dim=1)[0]
        

        encodedContentFeatures={}
        encodedStyleFeatures={}
        encodedContentCategory={}
        encodedStyleCategory={}
        
        encodedContentFeatures.update({'real': enc_content_list_onReal[-1]})
        encodedContentFeatures.update({'fake': contentFeaturesOnGenerated[-1]})
        encodedContentFeatures.update({'groundtruth': GT_content_outputs[-1]})
        
        encodedStyleFeatures.update({'real': reshaped_style_list_onReal[-1]})
        encodedStyleFeatures.update({'fake': styleFeaturesOnGenerated[-1]})
        encodedStyleFeatures.update({'groundtruth': GT_style_outputs[-1]})
        
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