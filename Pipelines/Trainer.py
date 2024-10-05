import logging
import os
import shutil
import sys
import numpy as np
import torch
torch.set_warn_always(False)
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from time import time
from tqdm import tqdm

import multiprocessing
import glob


import random
DISP_CONTENT_STYLE_NUM=3
LOG_INTERVAL=30


sys.path.append('./')
from Pipelines.Dataset import CASIA_Dataset
from Networks.utils.ops import create_if_not,set_random,unormalize
from Networks.GeneralizedGenerator.WNetGeneratorBase import WNetGeneratorBase as GeneralizedWNet
from Networks.Generators.PlainWNetBase import WNetGenerator as PlainWnet

from LossAccuracyEntropy.loss import WLoss

WNetDict = {'general':GeneralizedWNet,
            'plain': PlainWnet}


class Trainer(nn.Module):
    def __init__(self, hyperParams=-1, penalties=-1):
        super().__init__()
        # config
        self.config=hyperParams
        self.penalties=penalties
        
        self.debug = self.config.debug
        
        # for k,v in config.items():
        #     print(k,v)
        set_random()
        

        self.iter_num = 0
        self.start_epoch = 0
        self.best_epoch = 0
        self.best_val_l1loss = torch.tensor(float('inf')).cuda()
        
        

        # model
        WNetGenerator = WNetDict[self.config.wnet]
        
        self.model = WNetGenerator(self.config)
        # testContent = torch.randn(self.config.trainParams.batchSize, self.config.datasetConfig.inputContentNum, 64, 64).to('cuda')  
        # testStyle = torch.randn(self.config.trainParams.batchSize, self.config.datasetConfig.inputStyleNum, 64, 64).to('cuda')  
        # testGT=torch.randn(self.config.trainParams.batchSize, 1, 64, 64).to('cuda')  
        # self.model.forward(testContent, testStyle, testGT)
        # self.model = nn.DataParallel(WNetGenerator(config).cuda())
        self.model.train()
        #self.model.cuda(device=self.config.device[0])
        self.model.cuda()
        
        # xavier initialization
        for m in self.model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
        

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.trainParams.initLr , 
                                          weight_decay=self.penalties.generator_weight_decay_penalty)
        lrGamma = np.power(0.01, 1.0 / (self.config.trainParams.epochs - 1))
        self.scheculer = torch.optim.lr_scheduler.ExponentialLR(gamma=lrGamma,optimizer=self.optimizer)

        # loss 
        self.sumLoss = WLoss(self.config, self.penalties)

        # dataloader
        if not self.debug:
            num_workers = multiprocessing.cpu_count()//3
        else:
            num_workers=0
        # num_workers=0
        train_dataset = CASIA_Dataset(self.config)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.trainParams.batchSize, num_workers=num_workers,pin_memory=True, shuffle=True,drop_last=True)
        #self.train_loader = DataLoader(train_dataset, batch_size=self.batchsize, num_workers=0,pin_memory=True, shuffle=True,drop_last=True)
        val_dataset = CASIA_Dataset(self.config, is_train=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.trainParams.batchSize,    
                                     num_workers=num_workers, pin_memory=True, shuffle=False,drop_last=True)
        #self.val_loader = DataLoader(val_dataset, batch_size=self.val_batchsize,    num_workers=0, pin_memory=True, shuffle=False,drop_last=True)


        #logging and ckpt
        if self.config.userInterface.resumeTrain==0 and os.path.exists(self.config.userInterface.logDir):
            shutil.rmtree(self.config.userInterface.logDir)
        if self.config.userInterface.resumeTrain==0 and os.path.exists(self.config.userInterface.trainImageDir):
            shutil.rmtree(self.config.userInterface.trainImageDir)
        if self.config.userInterface.resumeTrain==0 and os.path.exists(self.config.userInterface.expDir):
            shutil.rmtree(self.config.userInterface.expDir)
        
        if not os.path.exists(self.config.userInterface.logDir):
            os.makedirs(self.config.userInterface.logDir)
        if not os.path.exists(self.config.userInterface.trainImageDir):
            os.makedirs(self.config.userInterface.trainImageDir)
        if not os.path.exists(self.config.userInterface.expDir):
            os.makedirs(self.config.userInterface.expDir)

        
        logging.basicConfig(filename=self.config.userInterface.logDir+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        self.writer = SummaryWriter(self.config.userInterface.logDir)


        # resume
        if self.config.userInterface.resumeTrain:
            logging.info('Load model from %s'% self.config.userInterface.expDir)
            list_of_files = glob.glob(self.config.userInterface.expDir+'/*.pth')
            latest_file =max(list_of_files, key=os.path.getctime)
            ckpt = torch.load(latest_file)
            
            #ckpt = torch.load(self.config.userInterface.expDir)
            self.start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            
            # for _ in range(self.loadedEpoch):
            #     self.scheculer.step()
            # self.start_epoch = self.loadedEpoch+1
            
        logging.info('Trainer prepared.') 

    
    def SummaryWriting(self, evalContents, evalStyles, evalGTs, evalFakes,  step, lossG, lossDict, mark='NA'):
        dispContentNum=min(DISP_CONTENT_STYLE_NUM, evalContents.shape[1])
        dispStyleNum=min(DISP_CONTENT_STYLE_NUM, evalStyles.shape[1])
        selectedContentIdx = random.sample(range(evalContents.shape[1]), dispContentNum)
        selectedStyleIdx = random.sample(range(evalStyles.shape[1]), dispStyleNum)
        out_grids = []
        for batch_id in range(self.config.trainParams.batchSize):
            decode_vis = evalFakes[-1][batch_id]
            GT_vis = evalGTs[batch_id]
            
            
            contents_vis = evalContents[batch_id][selectedContentIdx].unsqueeze(1)
            styles_vis = evalStyles[batch_id][selectedStyleIdx].unsqueeze(1)
            

            out_list=[]
            for x in contents_vis: out_list.append(x)
            for x in styles_vis: out_list.append(x)
            out_list.append(decode_vis)
            out_list.append(GT_vis)
            

            out_grid = make_grid([x for x in out_list], nrow=dispContentNum+dispStyleNum+2, normalize=True, scale_each=True)
            out_grids.append(out_grid) 
        output = make_grid(out_grids, nrow=1, normalize=True, scale_each=True)
        self.writer.add_image(mark+' Image', output , dataformats='CHW', global_step=step)
        
        # if not lossG==-1:
        self.writer.add_scalar('00-SumLossG-'+mark, lossG, global_step=step)
        self.writer.add_scalar('01-LossReconstruction/L1-'+mark, lossDict['l1_loss'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/ConstContent-'+mark, lossDict['const_content_loss'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/ConstStyle-'+mark, lossDict['const_style_loss'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryRealContent-'+mark, lossDict['content_category_OnOrg'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryFakeContent-'+mark, lossDict['content_category_OnGen'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryRealStyle-'+mark, lossDict['style_category_OnOrg'], global_step=step)
        self.writer.add_scalar('01-LossGenerator/CategoryFakeStyle-'+mark, lossDict['style_category_OnGen'], global_step=step)          
        self.writer.add_scalar('011-LossDeepPerceptual-ContentMSE-'+mark, lossDict['contentMSE'], global_step=step)
        self.writer.add_scalar('011-LossDeepPerceptual-StyleMSE-'+mark, lossDict['styleMSE'], global_step=step)
        self.writer.add_scalar('00-LearningRate', self.scheculer.get_lr()[0], global_step=step)
        self.trainStart = time()
    
    def TrainOneEpoch(self, epoch):
        time1 = time()
        trainTimeStart = time()
        for idx,(contents, styles, GT_style,content_onehot,style_onehot) in \
            tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Epoch: %d Training Progress" % epoch):
            contents, styles, GT_style,content_onehot,style_onehot = contents.cuda(), styles.cuda(), GT_style.cuda() ,content_onehot.cuda(),style_onehot.cuda()
            # reshape_contents = contents.reshape(self.batchsize*self.input_content_num,1,64,64)            
            reshape_styles = styles.reshape(self.config.trainParams.batchSize*self.config.datasetConfig.inputStyleNum,1,64,64)

            reshaped_content_list,content_category, \
            reshaped_enc_style_list, style_category, \
            decode_output_list,GT_output = self.forward(contents,reshape_styles,GT_style)

            sumLossG,Loss_dict = self.sumLoss(reshaped_content_list,
                                              content_category,
                                              reshaped_enc_style_list, 
                                              style_category,
                                              decode_output_list,
                                              GT_output,GT_style,
                                              content_onehot,style_onehot)

            self.optimizer.zero_grad()
            sumLossG.backward()
            self.optimizer.step()
            self.iter_num = self.iter_num + 1

            
            if time()-trainTimeStart>LOG_INTERVAL or idx==0 or idx == len(self.train_loader)-1:
                self.SummaryWriting(evalContents=contents, evalStyles=styles, evalGTs=GT_style, evalFakes=decode_output_list,  
                                    step=epoch*1000+int(idx/len(self.train_loader)*1000), lossG=sumLossG, lossDict=Loss_dict, mark='Train')
                trainTimeStart = time()

        
        time2 = time()
        logging.info('Epoch: %d training time :%f minutes' % (epoch, (time2-time1)/60))

    def TestOneEpoch(self,epoch):
        is_train = False
        self.model.eval()
        testStart = time()
        with torch.no_grad():
            for idx,(val_contents, val_styles, val_GT_style,val_content_onehot,val_style_onehot) in \
                tqdm(enumerate(self.val_loader), total=len(self.val_loader), desc="Epoch: %d Test at Training" % epoch):
                contents, styles, GT_style,content_onehot,style_onehot = val_contents.cuda(), val_styles.cuda(), val_GT_style.cuda() ,val_content_onehot.cuda(),val_style_onehot.cuda()
                reshape_styles = styles.reshape(self.config.trainParams.batchSize*self.config.datasetConfig.inputStyleNum,1,64,64)
                
                enc_content_list,content_category, \
                reshaped_enc_style_list, style_category, \
                decode_output_list,GT_output = self.forward(contents,reshape_styles,GT_style,is_train=is_train)
                
                
                sumLossG,Loss_dict = self.sumLoss(enc_content_list,
                                              content_category,
                                              reshaped_enc_style_list, 
                                              style_category,
                                              decode_output_list,
                                              GT_output,GT_style,
                                              content_onehot,style_onehot)
                
                
                if time()-testStart>LOG_INTERVAL//10 or idx==0 or idx == len(self.val_loader)-1:
                    self.SummaryWriting(evalContents=contents, evalStyles=styles, evalGTs=GT_style, evalFakes=decode_output_list,  
                                        step=epoch*1000+int(idx/len(self.val_loader)*1000), lossG=sumLossG, lossDict=Loss_dict, mark='Test')
                    testStart=time()
                

    def Pipelines(self):
        train_start = time()
        
        # eiStart = 
        training_epoch_list = range(self.start_epoch,self.config.trainParams.epochs,1)      
        self.trainStart = time()
        
        self.TestOneEpoch(0)
        for epoch in training_epoch_list:    
            self.TrainOneEpoch(epoch)
            self.TestOneEpoch(epoch+1)
            state = {
                'epoch':epoch+1,
                'state_dict':self.model.state_dict(),
                'optimizer':self.optimizer.state_dict(),
            }
            torch.save(state,self.config.userInterface.expDir+'/CkptEpoch%d.pth' % epoch+1)
            logging.info(f'save model at {epoch+1} epoch')
            self.scheculer.step()
        
        self.model.train()
        train_end = time()
        training_time = (train_end - train_start) / 3600
        self.writer.close()
        logging.info('Training finished, tensorboardX writer closed')
        logging.info('Training total time: %f hours.' % training_time)
        
    def forward(self,reshape_contents,reshape_styles,GT_style,is_train=True):
        enc_content_list,content_category, \
        reshaped_enc_style_list, style_category, \
            decode_output_list,GT_output = self.model(reshape_contents,reshape_styles,GT_style,is_train)
        return enc_content_list,content_category, \
            reshaped_enc_style_list, style_category, \
            decode_output_list,GT_output
           