import argparse
import logging
import os
import random
import shutil
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from time import time
from tqdm import tqdm


sys.path.append('./')
from Pipelines.Dataset import CASIA_Dataset
from Networks.utils.ops import create_if_not,set_random,unormalize
from Networks.Generators.PlainWNetBase import WNetGenerator
from LossAccuracyEntropy.loss import WLoss
from config import CONFIG as cfg

class see(nn.Module):
    def __init__(self,config):
        super().__init__()
        # config
        self.config=config
        # for k,v in config.items():
        #     print(k,v)
        set_random()
        # parameters
        self.in_channels = config['in_channels']
        self.generator_dim = config['generator_dim']
        self.batchsize = config['batch_size']
        self.val_batchsize = config['val_batch_size']
        self.input_content_num = config['inputContentNum']
        self.input_style_num = config['inputStyleNum']
        self.max_epoch = config['max_epoch']
        self.iter_num = 0
        self.start_epoch = 0
        self.best_epoch = 0

        # dataloader
        val_dataset = CASIA_Dataset(config,is_train=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.val_batchsize,num_workers=8,pin_memory=True, shuffle=False,drop_last=True)

        # model
        self.model = WNetGenerator(config)
        self.model.eval()
        self.model.cuda()

        #logging
        snapshot_path = config['log_dir']
        create_if_not(snapshot_path)
        self.save_model_path = snapshot_path + '/model'
        create_if_not(self.save_model_path)
        logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.writer = SummaryWriter(snapshot_path+'/tensorboard')

        # resume
        if config['resume']:
            logging.info('Load model from %s'%config['ckpt_path'])
            ckpt = torch.load(config['ckpt_path'])
            self.start_epoch = ckpt['epoch'] + 1
            self.model.load_state_dict(ckpt['state_dict'])
        logging.info('Trainer prepared.') 

    def validate(self,epoch):
        with torch.no_grad():
            for idx,(val_contents, val_styles, val_GT_style,val_content_onehot,val_style_onehot) in enumerate(self.val_loader):
                contents, styles, GT_style,content_onehot,style_onehot = val_contents.cuda(), val_styles.cuda(), val_GT_style.cuda() ,val_content_onehot.cuda(),val_style_onehot.cuda()
                # reshape_contents = contents.reshape(self.val_batchsize*self.input_content_num,1,64,64)
                reshape_styles = styles.reshape(self.val_batchsize*self.input_style_num,1,64,64)
                
                enc_content_list,content_category, \
                reshaped_enc_style_list, style_category, \
                decode_output_list,GT_output = self.forward(contents,reshape_styles,GT_style,is_train=False)
                
                out_grids = []
                for batch_id in range(self.val_batchsize):
                    decode_vis = decode_output_list[-1][batch_id]
                    GT_vis = GT_style[batch_id]
                    contents_vis = contents[batch_id][:7].unsqueeze(1)
                    styles_vis = styles[batch_id].unsqueeze(1)
                    out_list = [decode_vis,GT_vis]
                    for x in contents_vis: out_list.append(x)
                    for x in styles_vis: out_list.append(x)
                    # GT_style_grid = make_grid(GT_vis, nrow=1, normalize=True, scale_each=True)
                    # val_contents_grid = make_grid([x for x in contents_vis], nrow=64, normalize=True, scale_each=True)
                    # val_styles_grid = make_grid([x for x in styles_vis], nrow=5, normalize=True, scale_each=True)
                    out_grid = make_grid([x for x in out_list], nrow=70, normalize=False, value_range=(-1,1),scale_each=True)
                    out_grids.append(out_grid)
                random_numbers = random.sample(range(1500), 50)                
                if idx in random_numbers:
                    output = make_grid(out_grids, nrow=1, normalize=False)

                    self.writer.add_image(f'output_{idx}', output  , epoch,dataformats='CHW')

    def do_see(self):

        for epoch in range(self.start_epoch,self.max_epoch):
            self.validate(epoch)
            break

        self.writer.close()

    def forward(self,reshape_contents,reshape_styles,GT_style,is_train=True):
        enc_content_list,content_category, \
        reshaped_enc_style_list, style_category, \
        decode_output_list,GT_output = self.model(reshape_contents,reshape_styles,GT_style,is_train)
        return enc_content_list,content_category, \
            reshaped_enc_style_list, style_category, \
            decode_output_list,GT_output
           
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
    parser.add_argument('--resume',type=bool,default=False,help='resume or not')
    parser.add_argument('--log_dir',type=str,default='./log/see_0703_',help='log dir')
    parser.add_argument('--ckpt_path',type=str,default='log/0703_full/model/checkpoint_best.pth',help='checkpoint path')
    args = parser.parse_args() 

    cfg['resume'] = args.resume
    cfg['log_dir'] = args.log_dir
    cfg['ckpt_path'] = args.ckpt_path  
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    #test
    # cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
    # cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
    # cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml' 
    # cfg['val_GT_style_yaml'] = 'cmy/test_list/val_GT_dir.yaml'
    # cfg['val_reference_style_yaml'] = 'cmy/test_list/val_reference_style_dir.yaml' 
    # cfg['batch_size'] = 8
    cfg['val_batch_size'] = 32
    model = see(cfg)
    model.do_see()