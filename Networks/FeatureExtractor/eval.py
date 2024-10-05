import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from FeatureExtractorBase import FeatureExtractor
from Dataset import Feature_Dataset
from tqdm import tqdm
sys.path.append('./')
from config import CONFIG as cfg
from Networks.utils.ops import set_random,cv2torch,string2tensor,read_file_to_dict


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if cfg['type'] == 'content':
        outputnum = 3755     
        type = cfg['type']
        order_txt = cfg['Label0_list']  
    elif cfg['type'] == 'style':
        outputnum = 80    
        type = cfg['type']
        order_txt = cfg['Label1_list'] 

    # 读取对应字典
    order = read_file_to_dict(order_txt) 
    # 初始化模型
    model = FeatureExtractor(output_nums=outputnum).to(device)  
    model.train()
    if cfg['type'] == 'content':
        ckpt_path = 'Networks/FeatureExtractor/ckpt/content_best_extractor_model.pth'
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))

    elif cfg['type'] == 'style':
        ckpt_path = 'Networks/FeatureExtractor/ckpt/style_best_extractor_model.pth'
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))

    val_dataset = Feature_Dataset(cfg, type=type,is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    wrong_num,whole_num = 0,len(val_loader)
    for image, label,raw_label in tqdm(val_loader):
        image, label = image.to(device).float(), label.to(device)
        with torch.no_grad():
            output,_ = model(image)
            output = torch.nn.functional.softmax(output, dim=1)
    
             # 步骤4: 解释预测结果
            value, predicted_class = torch.max(output.cpu(), dim=1)
            predicted_class = list(order.keys())[predicted_class]
            
            # 输出预测结果和概率
            # print("Predicted class:", predicted_class)
            # print(f"raw class:{raw_label[0]}")
            if predicted_class == raw_label[0]:
                print('Right')
            else:
                print('Wrong')
                wrong_num += 1

    print(f'一共有{wrong_num/whole_num:.2f}的错误')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
    parser.add_argument('--ckpt_path',type=str,default='Networks/FeatureExtractor/ckpt',help='checkpoint path')
    parser.add_argument('--type',type=str,default='style',help='')
    args = parser.parse_args()
    cfg['gpu'] = args.gpu
    cfg['ckpt_path'] = args.ckpt_path
    cfg['type'] = args.type

    cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
    cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml' 
    cfg['val_GT_style_yaml'] = 'cmy/test_list/val_GT_dir.yaml'
    cfg['val_reference_style_yaml'] = 'cmy/test_list/val_reference_style_dir.yaml' 
    cfg['batch_size'] = 8
    cfg['val_batch_size'] = 8
    # 指定GPU编号
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']
    main(cfg)