import os
import sys
import torch
torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np

sys.path.append('./')
from Pipelines.Dataset import CASIA_Dataset
from Networks.FeatureExtractor.FeatureExtractorBase import FeatureExtractorBase as FeatureExtractor
from Networks.Generators.PlainWNetBase import WNetGenerator
from config import CONFIG as cfg
HighLevelFeaturePenaltyPctg=[0.1,0.15,0.2,0.25,0.3]

class WLoss(nn.Module):
    def __init__(self, config, penalty):
        super(WLoss, self).__init__()

        # penalities
        self.Pixel_Reconstruction_Penalty = penalty['Pixel_Reconstruction_Penalty']
        self.Lconst_content_Penalty = penalty['Lconst_content_Penalty']
        self.Lconst_style_Penalty = penalty['Lconst_style_Penalty']
        self.Generator_Categorical_Penalty = penalty['Generator_Categorical_Penalty']
        self.FeatureExtractorPenalty_ContentPrototype = penalty['FeatureExtractorPenalty_ContentPrototype']
        self.FeatureExtractorPenalty_StyleReference = penalty['FeatureExtractorPenalty_StyleReference']
        # self.HighLevelFeaturePenaltyPctg = config['HighLevelFeaturePenaltyPctg']
        # FeatureExtractor
        
        # if len(config.device)==2:
        #     featureExtractorDevice=config.device[1]
        # elif len(config.device)==1:
        #     featureExtractorDevice=config.device[0]
       
        self.contentExtractorList=[]
        for contentExtractor in config['extractorContent']:
            thisContentExtractor = FeatureExtractor(outputNums=len(config.datasetConfig.loadedLabel0Vec), 
                                                    modelSelect=contentExtractor.name,
                                                    type='content').extractor
            thisContentExtractor.eval()
            thisContentExtractor.cuda()
            
            # loaded = torch.load(contentExtractor.path)
            # new=list(loaded.items())
            # my_model_kvpair=thisContentExtractor.state_dict()
            # count=0
            # for key,value in my_model_kvpair.items():
            #     layer_name,weights=new[count]      
            #     my_model_kvpair[key]=weights
            #     count+=1
            # thisContentExtractor.load_state_dict(my_model_kvpair)
            self.NameMappingLoading(thisContentExtractor, contentExtractor.path)
            self.contentExtractorList.append(thisContentExtractor)


            
            # thisContentExtractor.load_state_dict(torch.load(contentExtractor.path), strict=False)
        
        self.styleExtractorList=[]
        for styleExtractor in config['extractorStyle']:
            thisStyleExtractor = FeatureExtractor(outputNums=len(config.datasetConfig.loadedLabel1Vec), 
                                                    modelSelect=styleExtractor.name,
                                                    type='style').extractor
            thisStyleExtractor.eval()
            thisStyleExtractor.cuda()
            # thisStyleExtractor.load_state_dict(torch.load(styleExtractor.path), strict=False)
            self.NameMappingLoading(thisStyleExtractor, styleExtractor.path)
            self.styleExtractorList.append(thisStyleExtractor)
        # a=1
        
        # self.content_extractor = FeatureExtractor(output_nums=len(config.datasetConfig.loadedLabel0Vec), )
        # self.content_extractor.eval()
        # self.content_extractor.cuda()
        # self.content_extractor_ckpt = config['extractorContent'][0].path
        # for contentPath in config['extractorContent']:
        #     self.content_extractor.load_state_dict(torch.load(self.content_extractor_ckpt))
        
        # if os.path.exists(content_extractor_ckpt):
        #     self.content_extractor.load_state_dict(torch.load(self.content_extractor_ckpt))
        # else: 
        #     print('No pretrained content extractor')
        #     return
        
        # self.style_extractor = FeatureExtractor(output_nums=len(config.datasetConfig.loadedLabel1Vec))
        # # self.style_extractor = FeatureExtractor(output_nums=80)
        # self.style_extractor.eval()
        # self.style_extractor.cuda()
        # self.style_extractor_ckpt = config['extractorStyle'][0].path
        # if os.path.exists(self.style_extractor_ckpt):
        #     self.style_extractor.load_state_dict(torch.load(self.style_extractor_ckpt))
        # else: 
        #     print('No pretrained content extractor')
        #     return

        
    def NameMappingLoading(self,extractor, path):
        loaded = torch.load(path)
        loadedItems=list(loaded.items())
        thisExtractorDict=extractor.state_dict()
        count=0
        for key,value in thisExtractorDict.items():
            layer_name,weights=loadedItems[count]      
            thisExtractorDict[key]=weights
            count+=1
        extractor.load_state_dict(thisExtractorDict)
        print(path+" loaded")
        
            

    def generator_loss(self, reshaped_content_list,content_category,\
                        reshaped_style_list, style_category,\
                        decode_output_list,GT_output, GT,content_onehot,style_onehot):
        # l1 loss
        l1_loss = F.l1_loss(decode_output_list[-1], GT, reduction='mean')

        # const_content loss
        GT_content_enc = GT_output[0]
        const_content_loss = F.mse_loss(reshaped_content_list[-1],GT_content_enc)
        # const_style loss
        GT_style_enc = GT_output[1]
        reshaped_styles = reshaped_style_list[-1]
        reshaped_styles = reshaped_styles.permute(1,0,2,3,4)
        style_losses = [F.mse_loss(enc_style, GT_style_enc) for enc_style in reshaped_styles]
        const_style_loss = torch.mean(torch.stack(style_losses))

        # category loss
        content_category_OnOrg,content_category_OnGen = 0,0
        GT_content_category = GT_output[2]
        for GT_logits,fake_logits,onehot in zip(content_category,GT_content_category,content_onehot):
            GT_logits = torch.nn.functional.softmax(GT_logits, dim=0)
            content_category_OnOrg += F.cross_entropy(GT_logits, onehot)
            fake_logits = torch.nn.functional.softmax(fake_logits, dim=0)            
            content_category_OnGen += F.cross_entropy(fake_logits, onehot)
        content_category_OnOrg /= len(GT_content_category)
        content_category_OnGen /= len(GT_content_category)
        
        style_category_OnOrg,style_category_OnGen = 0,0
        GT_style_category = GT_output[3]
        for GT_logits,fake_logits,onehot in zip(style_category,GT_style_category,style_onehot):
            GT_logits = torch.nn.functional.softmax(GT_logits, dim=0)
            style_category_OnOrg += F.cross_entropy(GT_logits, onehot)
            fake_logits = torch.nn.functional.softmax(fake_logits, dim=0)            
            style_category_OnGen += F.cross_entropy(fake_logits, onehot)
        style_category_OnOrg /= len(GT_style_category)
        style_category_OnGen /= len(GT_style_category)
        return l1_loss,const_content_loss,const_style_loss,content_category_OnOrg,content_category_OnGen,style_category_OnOrg,style_category_OnGen

    def FeatureExtractor_loss(self,GT,imgFake):
        # content_extractor
        contentMSE=0.0
        for idx, thisContentExtractor in enumerate(self.contentExtractorList):
            thisContentMSE=0
            with torch.no_grad():
                _,GT_content_features = thisContentExtractor(GT)
                _,fake_content_features = thisContentExtractor(imgFake)
            if not len(HighLevelFeaturePenaltyPctg) == \
                    len(GT_content_features) == \
                    len(fake_content_features):
                print('content length not paired')
                return
            for idx,(GT_content_feature,fake_content_feature) in enumerate(zip(GT_content_features,fake_content_features)):
                thisContentMSE += F.mse_loss(GT_content_feature,fake_content_feature)*self.FeatureExtractorPenalty_ContentPrototype[0]*HighLevelFeaturePenaltyPctg[idx]
            thisContentMSE /= len(HighLevelFeaturePenaltyPctg)
            contentMSE+=thisContentMSE*self.FeatureExtractorPenalty_ContentPrototype[idx]
        contentMSE = contentMSE / len(self.contentExtractorList)


        # style_extractor
        styleMSE=0.0
        for idx, thsiStyleExtractor in enumerate(self.styleExtractorList):
            thisStyleMSE = 0
            with torch.no_grad():
                _,GT_style_features = self.styleExtractorList[0](GT)
                _,fake_style_features = self.styleExtractorList[0](imgFake)
            if not len(HighLevelFeaturePenaltyPctg) == \
                    len(GT_style_features) == \
                    len(fake_style_features):
                print('style length not paired')
                return
            for idx,(GT_style_feature,fake_style_feature) in enumerate(zip(GT_style_features,fake_style_features)):
                thisStyleMSE += F.mse_loss(GT_style_feature,fake_style_feature)*self.FeatureExtractorPenalty_StyleReference[0]*HighLevelFeaturePenaltyPctg[idx]
            thisStyleMSE /= len(HighLevelFeaturePenaltyPctg)
            styleMSE+=thisStyleMSE
        styleMSE = styleMSE / len(self.styleExtractorList)

        return contentMSE,styleMSE



    def forward(self, reshaped_enc_content_list,content_category,\
                    reshaped_enc_style_list, style_category,\
                    decode_output_list,GT_output,GT,content_onehot,style_onehot):
        # generator_const_loss
        l1_loss,const_content_loss,const_style_loss,content_category_OnOrg,content_category_OnGen,style_category_OnOrg,style_category_OnGen = \
            self.generator_loss(reshaped_enc_content_list,content_category,\
                                reshaped_enc_style_list, style_category,\
                                decode_output_list,GT_output, GT,content_onehot,style_onehot)
        # generator_category_loss
        contentMSE,styleMSE = self.FeatureExtractor_loss(GT=GT,imgFake=decode_output_list[-1])

        if const_content_loss <= 1e8: self.Lconst_content_Penalty = 0.
        if const_style_loss <= 1e8: self.Lconst_style_Penalty = 0.        
        # sumLossG = l1_loss * self.Pixel_Reconstruction_Penalty + \
        #             const_content_loss * self.Lconst_content_Penalty + \
        #             const_style_loss * self.Lconst_style_Penalty + \
        #             (content_category_OnOrg + content_category_OnGen + \
        #              style_category_OnOrg + style_category_OnGen)* self.Generator_Categorical_Penalty + \
        #             contentMSE + \
        #             styleMSE
        sumLossG = l1_loss * self.Pixel_Reconstruction_Penalty + \
                    const_content_loss * self.Lconst_content_Penalty + \
                    const_style_loss * self.Lconst_style_Penalty 


        return sumLossG,{'l1_loss':l1_loss,
                         'const_content_loss':const_content_loss,
                         'const_style_loss':const_style_loss,
                         'content_category_OnOrg':content_category_OnOrg,
                         'content_category_OnGen':content_category_OnGen,
                         'style_category_OnOrg':style_category_OnOrg,
                         'style_category_OnGen':style_category_OnGen,
                         'contentMSE':contentMSE,
                         'styleMSE':styleMSE}

if __name__ == '__main__':
    cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
    cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml'

    batchsize = cfg['batchsize']
     # 创建CASIA数据集实例
    casia_dataset = CASIA_Dataset(cfg)
    # 创建DataLoader
    casia_loader = DataLoader(casia_dataset, batch_size=batchsize, shuffle=False,drop_last=True)
    Wnet = WNetGenerator(cfg)
    loss = WLoss(cfg)
    # 读入第一个样本
    for contents, styles, GT_style in casia_loader:
        contents, styles, GT_style = contents.cuda(), styles.cuda(), GT_style.cuda() 
        # reshape_contents = contents.reshape(batchsize*64, 1, 64, 64)

        reshape_styles = styles.reshape(batchsize*5,1,64,64)

        enc_content_list,content_category, \
        reshaped_enc_style_list, style_category, \
        decode_output_list,GT_output = Wnet(contents,reshape_styles,GT_style)

        sumLossG,Loss_dict = loss(enc_content_list,reshaped_enc_style_list,decode_output_list,GT_output,GT_style)
        print(Loss_dict)