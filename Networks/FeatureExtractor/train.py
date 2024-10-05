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


def main(cfg):
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if cfg['type'] == 'content':
        cfg['inputContentNum']=64
        cfg['Label0_list']= '/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label0.txt'
        outputnum = 3755     
        type = cfg['type']
    elif cfg['type'] == 'style':
        outputnum = 80    
        type = cfg['type']
        
        

    # 加载数据集
    train_dataset = Feature_Dataset(cfg, type=type)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

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


    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 初始化最佳loss为正无穷
    best_loss = float('inf')
    num_epochs = cfg['max_epoch']

    # 训练模型
    for epoch in range(num_epochs):
        running_loss = 0.0
        for idx, (images, labels ) in tqdm(enumerate(train_loader), total=len(train_loader),desc='Training Epoch: %d/%d' % (epoch+1, num_epochs)):
            images, labels = images.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            
            outputs,_ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # 计算平均loss
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}')
    
        # 如果当前loss是最佳的，保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(cfg['ckpt_path'], f'{type}_best_extractor_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved Best Model at Epoch {epoch+1} with Loss: {best_loss}')

    # 保存最后一个epoch的模型
    final_model_path = os.path.join(cfg['ckpt_path'], f'{type}_final_extractor_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved Final Model at Epoch {num_epochs}')

    # 输出最终的loss
    print(f'Final Loss: {avg_loss}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
    parser.add_argument('--ckpt_path',type=str,default='Networks/FeatureExtractor/ckpt',help='checkpoint path')
    parser.add_argument('--type',type=str,default='style',help='')
    
    args = parser.parse_args()
    cfg['gpu'] = args.gpu
    cfg['ckpt_path'] = args.ckpt_path
    cfg['type'] = args.type

    cfg['content_yaml'] = 'cmy/CASIA/content_dir.yaml'
    cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml' 
    cfg['val_GT_style_yaml'] = 'cmy/test_list/val_GT_dir.yaml'
    cfg['val_reference_style_yaml'] = 'cmy/test_list/val_reference_style_dir.yaml' 
    # cfg['batch_size'] = 8
    # cfg['val_batch_size'] = 8
    cfg['max_epoch']=1000
    # 指定GPU编号
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']
    main(cfg)
