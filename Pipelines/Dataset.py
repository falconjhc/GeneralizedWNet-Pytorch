import argparse
import os
import random
import sys
import numpy as np
import torch
torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
import cv2
from time import time
sys.path.append('./')
from Networks.utils.ops import set_random,cv2torch,read_file_to_dict
from config import CONFIG as cfg
import time
from tqdm import tqdm
displayInterval = 25

def find_label0(file_name):
    return file_name.split('_')[1]

class CASIA_Dataset(Dataset):
    def __init__(self,config,is_train=True,
                #  content_dir='CASIA_Dataset/PrintedData_64Fonts/Simplified/GB2312_L1/',
                #  style_dir='CASIA_Dataset/PrintedData/GB2312_L1/',
                #  base_dir='/data0/haochuan/'
                 ):
        
        # self.dataConfig = config['dataConfig']
        if is_train:
            self.GT_style_yaml = os.path.join(config.datasetConfig.yamls, 'TrainGroundTruth.yaml')
            self.reference_style_yaml = os.path.join(config.datasetConfig.yamls, 'TrainStyleReference.yaml')
        else:
            self.GT_style_yaml = os.path.join(config.datasetConfig.yamls, 'TestGroundTruth.yaml')
            self.reference_style_yaml = os.path.join(config.datasetConfig.yamls, 'TestStyleReference.yaml')
            
        self.content_yaml = os.path.join(config.datasetConfig.yamls, 'Content.yaml')            
        
        
        self.input_content_num = config.datasetConfig.inputContentNum
        self.input_style_num = config.datasetConfig.inputStyleNum 
        set_random()
        

        # if not is_train:
        #     self.GT_style_yaml = config['val_GT_style_yaml']    
        #     self.reference_style_yaml = config['val_reference_style_yaml']

        strat_time = time.time()
        self.GT_data_list = self.data_list_create(self.GT_style_yaml)
        
        with open(self.content_yaml, 'r', encoding='utf-8') as f:
            print("Loading "+ self.content_yaml + '...', end='\r')
            content_files = yaml.load(f.read(), Loader=yaml.FullLoader)
            print("Loading "+ self.content_yaml + ' completed.')
        
        with open(self.reference_style_yaml, 'r', encoding='utf-8') as f:
            print("Loading "+ self.reference_style_yaml + '...', end='\r')
            reference_style_files = yaml.load(f.read(), Loader=yaml.FullLoader)
            print("Loading "+ self.reference_style_yaml + ' completed.')
        
        self.content_list,self.reference_style_list = [],[]
        
        
        # timeStart = time.time()
        # counter = 0
        for idx, (_, label0, label1) in tqdm(enumerate(self.GT_data_list), total=len(self.GT_data_list), desc="Loading: "):
        #for _,label0,label1 in self.GT_data_list:
            # if counter ==1 or time.time()-timeStart>displayInterval or counter == len(self.GT_data_list)-1:
            #     timeStart = time.time()
            #     print("%d %d" % (counter, len(self.GT_data_list)))
            # counter +=1
            # contetn_prefix = os.path.join(base_dir,content_dir)
            content_files[label0] = [path for path in content_files[label0]]
            self.content_list.append(random.sample(content_files[label0], self.input_content_num))
            
            # style_prefix = os.path.join(base_dir,style_dir)
            reference_style_files[label1] = [path for path in reference_style_files[label1]]
            self.reference_style_list.append(random.sample(reference_style_files[label1], self.input_style_num))
        
        # label0_order_txt = cfg['Label0_list']
        self.label0order = config.datasetConfig.loadedLabel0Vec
        
        # label1_order_txt = cfg['Label1_list']
        self.label1order = config.datasetConfig.loadedLabel1Vec

        self.content_onehot,self.style_onehot = [0 for _ in range(len(self.label0order))],[0 for _ in range(len(self.label1order))]
        end_time = time.time()
        print(f'dataset cost:{(end_time-strat_time):.2f}s')

    def __getitem__(self, index):
        transform1 = transforms.Compose([
            transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
            transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
        ])
        
        transform2 = transforms.Compose([
            
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=(64,64), scale=(0.8,1.0), antialias=True),
        ])
        
        transform = transforms.Compose([
            
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomResizedCrop(size=(64,64), scale=(0.8,1.0), antialias=True),
            transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
            transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1]
        ])
        
        content_tensors = torch.cat([cv2torch(content,transform1) for content in self.content_list[index]], dim=0)
        content = self.content_list[index][0][:-4].split('_')[-2]
        content = self.label0order[content]
        content_onehot =  torch.tensor(self.content_onehot)
        content_onehot[content] = 1
        
        GT_style_image = cv2torch(self.GT_data_list[index][0],transform1)
        GT_and_content=torch.cat((content_tensors, GT_style_image), 0)
        GT_and_content = transform2(GT_and_content)
        content_tensors = GT_and_content[:-1,:,:]
        GT_style_image= torch.unsqueeze(GT_and_content[-1,:,:],0)

        
        
        reference_style_tensors = torch.cat([cv2torch(reference_style,transform) for reference_style in self.reference_style_list[index]],dim=0)
        style = self.reference_style_list[index][0][:-4].split('_')[-1]
        while style[0]=='0' and len(style)>1:
            style=style[1:]
        style = self.label1order[style]
        style_onehot =  torch.tensor(self.style_onehot)
        style_onehot[style] = 1
        
        
        
        
        
        
        

        # return content_tensors.float().to(self.device), \
        #     reference_style_tensors.float().to(self.device),\
        #         GT_style_image.float().to(self.device),\
        #             content_onehot.float().to(self.device),\
        #                 style_onehot.float().to(self.device)  # 确保返回图像和标签
        
        return content_tensors.float(), \
            reference_style_tensors.float(),\
                GT_style_image.float(),\
                    content_onehot.float(),\
                        style_onehot.float()  # 确保返回图像和标签


    def __len__(self):
        return len(self.GT_data_list) 

    def data_list_create(self,yaml_name):
        data_list = []
        with open(yaml_name, 'r', encoding='utf-8') as f:
            print("Loading "+ yaml_name + '...', end='\r')
            iteration_files = yaml.load(f.read(), Loader=yaml.FullLoader)
            print("Loading "+ yaml_name + ' completed.')
            # print("YAML of GroundTruth Loaded")
       #  timeStart=time.time()
        counter=0
        for idx, (k,values) in tqdm(enumerate(iteration_files.items()), total=len(iteration_files.items()), desc="Test"):
            counter+=1
            # if time.time()-timeStart>displayInterval:
            #     timeStart=time.time()
            #     print("%d %d" % (counter, len(iteration_files.items())))
            path, label0,label1 = values
            # data_path = os.path.join(base_dir,input_dir)
            #file_path = os.path.join(data_path,k)
            data_list.append((path, label0,label1))
        # print("Dataset List Created")
        return data_list


if __name__ == "__main__":
    cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
    cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml' 
    cfg['val_GT_style_yaml'] = 'cmy/test_list/val_GT_dir.yaml'
    cfg['val_reference_style_yaml'] = 'cmy/test_list/val_reference_style_dir.yaml' 
    cfg['batch_size'] = 8
    cfg['val_batch_size'] = 8

    # 创建CASIA数据集实例
    casia_dataset = CASIA_Dataset(cfg)

    # 创建DataLoader
    casia_loader = DataLoader(casia_dataset, batch_size=8, shuffle=False)

    # 读入第一个样本
    for contents, styles, GT_style,content_onehot,style_onehot in casia_loader:
        
        print(contents.shape)
        print(styles.shape)

        print(GT_style.shape)
