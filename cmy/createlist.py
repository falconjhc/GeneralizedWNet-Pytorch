import sys
import os
import numpy as np
from tqdm import tqdm
import yaml

dataPathRoot= '/data0/haochuan/'
content_dir = 'CASIA_Dataset/PrintedData/GB2312_L1'
style_dir = 'CASIA_Dataset/PrintedData/GB2312_L1/'
content_list = 'FileList/PrintedData/Char_0_3754_64PrintedFonts_GB2312L1_Simplified.txt'
train_list= 'TrainTestFileList/PrintedData/Char_0_3754_Font_0_49_GB2312L1_Train.txt'
val_list= 'TrainTestFileList/PrintedData/Char_0_3754_Font_50_79_GB2312L1_Test.txt'
label0 = '/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label0.txt'
label1 = '/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label1.txt'

def readtxt(path):
    res = []
    f = open(path,'r')
    lines = f.readlines()
    for ll in lines:
        res.append(ll.replace('\n',''))
    f.close()
    return res

if __name__ == '__main__':
    label0_list = readtxt(label0)
    label1_list = readtxt(label1)
    dataset = readtxt(train_list)
    # dataset = readtxt(val_list)
    # dataset = readtxt(content_list)
    content_dir = {content:[] for content in label0_list}
    style_dir = {str(int(index)):[] for index in label1_list}
    iteration_dir = {x.split("@")[3]:[] for x in dataset}

    for x in tqdm(dataset):
        splits = x.split("@")
        content = splits[1]
        style = splits[2]
        name = splits[3]
        if content in label0_list:
            content_dir[content].append(name)
        if style in style_dir.keys():    
            style_dir[style].append(name)
        iteration_dir[name] = [content,style]     
    print("Save trainset")
    # 将content_dir保存为YAML文件
    # with open('cmy/list/train_content_dir.yaml', 'w') as file:
    #     yaml.dump(content_dir, file, allow_unicode=True)
    # 将style_dir保存为YAML文件
    with open('cmy/list/train_reference_style_dir.yaml', 'w') as file:
        yaml.dump(style_dir, file, allow_unicode=True)
    with open('cmy/list/train_GT_dir.yaml', 'w') as file:
        yaml.dump(iteration_dir, file, allow_unicode=True)



    dataset = readtxt(val_list)
    content_dir = {content:[] for content in label0_list}
    style_dir = {str(int(index)):[] for index in label1_list}
    iteration_dir = {x.split("@")[3]:[] for x in dataset}
    for x in tqdm(dataset):
        splits = x.split("@")
        content = splits[1]
        style = splits[2]
        name = splits[3]
        if content in label0_list:
            content_dir[content].append(name)
        if style in style_dir.keys():    
            style_dir[style].append(name)
        iteration_dir[name] = [content,style]   
    print("Save valset")
    # # 将content_dir保存为YAML文件
    # with open('cmy/list/val_content_dir.yaml', 'w') as file:
    #     yaml.dump(content_dir, file, allow_unicode=True)
    # 将style_dir保存为YAML文件
    with open('cmy/list/val_reference_style_dir.yaml', 'w') as file:
        yaml.dump(style_dir, file, allow_unicode=True)
    with open('cmy/list/val_GT_dir.yaml', 'w') as file:
        yaml.dump(iteration_dir, file, allow_unicode=True)

    dataset = readtxt(content_list)
    content_dir = {content:[] for content in label0_list}
    style_dir = {str(int(index)):[] for index in label1_list}
    iteration_dir = {x.split("@")[3]:[] for x in dataset}

    for x in tqdm(dataset):
        splits = x.split("@")
        content = splits[1]
        style = splits[2]
        name = splits[3]
        if content in label0_list:
            content_dir[content].append(name)
        if style in style_dir.keys():    
            style_dir[style].append(name)
        iteration_dir[name] = [content,style]   
    print("Save prototypeset")
    # 将content_dir保存为YAML文件
    with open('cmy/list/content_dir.yaml', 'w') as file:
        yaml.dump(content_dir, file, allow_unicode=True)
    # # 将style_dir保存为YAML文件
    # with open('cmy/list/prototype_style_dir.yaml', 'w') as file:
    #     yaml.dump(style_dir, file, allow_unicode=True)
    # with open('cmy/list/content_ iteration_dir.yaml', 'w') as file:
    #     yaml.dump(iteration_dir, file, allow_unicode=True)
