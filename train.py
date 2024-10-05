import argparse
import os



dataPathRoot= '/data0/haochuan/'
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()

from Pipelines.Trainer import Wnet
from config import CONFIG as cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,  default='3', help='GPU to use')
    parser.add_argument('--resume',type=int,default=False,help='resume or not')
    parser.add_argument('--log_dir',type=str,default='./log/lastexp',help='log dir')
    parser.add_argument('--ckpt_path',type=str,default='',help='checkpoint path')
    args = parser.parse_args() 

    cfg['resume'] = args.resume
    cfg['log_dir'] = args.log_dir
    cfg['ckpt_path'] = args.ckpt_path  
    cfg['in_channels']= 1
    cfg['generator_dim']= 64
    
    
    # cfg['dataConfig']='YamlLists/Debug/'
    # cfg['Label0_list']='/data0/haochuan/CASIA_Dataset/LabelVecs/HW300-Label0.txt'
    # cfg['Label1_list']='/data0/haochuan/CASIA_Dataset/LabelVecs/HW300-Label1.txt'
    # cfg['inputContentNum']=5
    # cfg['inputStyleNum']=3
    
    
    cfg['dataConfig']='YamlListsTest/PF64-PF80/'
    cfg['Label0_list']='/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label0.txt'
    cfg['Label1_list']='/data0/haochuan/CASIA_Dataset/LabelVecs/PF80-Label1.txt'
    cfg['inputContentNum']=64
    cfg['inputStyleNum']=3
    
    
    cfg['content_extractor_ckpt'] ='/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/content_best_extractor_model.pth'
    cfg['style_extractor_ckpt'] ='/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/style_best_extractor_model.pth'
    
    
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #test
    # cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
    # cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    # cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml' 
    # cfg['val_GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    # cfg['val_reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml' 
    # cfg['batch_size'] = 8
    # cfg['val_batch_size'] = 8

    model = Wnet(cfg)
    model.do_train()
