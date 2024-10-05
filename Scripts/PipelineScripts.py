# # -*- coding: utf-8 -*-
# from __future__ import print_function
# from __future__ import absolute_import

import argparse
import os
import sys
sys.path.append('../')
os.chdir(sys.path[0])

# os.system('cls' if os.name == 'nt' else 'clear')

dataPathRoot= '/data0/haochuan/'
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import shutup
shutup.please()

# from config import CONFIG as cfg






from Pipelines.Trainer import Trainer
from Configurations.ConfigurationOrganization import ParameterSetting  # noqa: E402
eps = 1e-9

import importlib
from easydict import EasyDict

# OPTIONS SPECIFICATION
# resumeTrain   =   0: training from stratch
#                   1: training from a based model
parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--config', dest='config',type=str,required=True)
parser.add_argument('--resumeTrain', dest='resumeTrain', type=int,required=True)
parser.add_argument('--initLr', dest='initLr',type=float,required=True)
parser.add_argument('--batchSize', dest='batchSize', type=int,required=True)
parser.add_argument('--epochs', dest='epochs', type=int, required=True)
parser.add_argument('--skipTest', dest='skipTest', type=bool, default=False)
parser.add_argument('--encoder', dest='encoder', type=str, required=False)
parser.add_argument('--mixer', dest='mixer', type=str, required=True)
parser.add_argument('--decoder', dest='decoder', type=str, required=False, default=None)
parser.add_argument('--device', dest='device', type=str, required=True, default='cpu')
parser.add_argument('--debug', dest='debug', type=int, required=False, default=0)

parser.add_argument('--wnet', dest='wnet', type=str, required=True, default='general')



# def main(_):
    

if __name__ == '__main__':
    args = parser.parse_args()
    hyperParams = \
    ParameterSetting(EasyDict(importlib.import_module('.'+args.config, 
                                                      package='Configurations').hyperParams), args).config
    penalties = EasyDict(importlib.import_module('.'+args.config, 
                                                 package='Configurations').penalties)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    print("#####################################################")
    model = Trainer(hyperParams=hyperParams, penalties=penalties)
    model.Pipelines()
