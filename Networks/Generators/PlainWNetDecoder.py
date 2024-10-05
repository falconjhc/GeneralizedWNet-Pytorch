import sys
# sys.path.append('./Networks')
import torch
import torch.nn as nn
import numpy as np


from Networks.utils.ops import Deconvblock

class Decoder(nn.Module):
    def __init__(self, normalization ,out_channels=1,generator_dim = 64):
        super(Decoder, self).__init__()
        # self.config = config
        self.normalization = normalization
        
        #1024*1*1
        self.decoderblock_one = Deconvblock(generator_dim*8*2, generator_dim*8, normalization=normalization) #512
        self.decoderblock_two = Deconvblock(generator_dim*8*3, generator_dim*8, normalization=normalization) #512
        self.decoderblock_three = Deconvblock(generator_dim*8*3, generator_dim*4, normalization=normalization) #256
        self.decoderblock_four = Deconvblock(generator_dim*4*3, generator_dim*2, normalization=normalization) #128
        self.decoderblock_five = Deconvblock(generator_dim*2*2, generator_dim, normalization=normalization) #64
        self.decoderblock_six = Deconvblock(generator_dim*2, out_channels, normalization='none',dropout=False) #1



    def forward(self, inputs_mix):
        input1,input2,input3,input4,input5,input6 = inputs_mix[::-1]
        de_x1 = self.decoderblock_one(input1)

        x1 = torch.concat((de_x1,input2),dim=1)
        de_x2 = self.decoderblock_two(x1)

        x2 = torch.concat((de_x2,input3),dim=1)
        de_x3 = self.decoderblock_three(x2)

        x3 = torch.concat((de_x3,input4),dim=1)
        de_x4 = self.decoderblock_four(x3)

        x4 = torch.concat((de_x4,input5),dim=1)
        de_x5 = self.decoderblock_five(x4)

        x5 = torch.concat((de_x5,input6),dim=1)
        de_x6 = self.decoderblock_six(x5)
        de_x6 = torch.tanh(de_x6)
        # de_x6 = torch.sigmoid(de_x6)

        
        # print(de_x1.shape)
        # print(de_x2.shape)
        # print(de_x3.shape)
        # print(de_x4.shape)
        # print(de_x5.shape)
        # print(de_x6.shape)
        res = [de_x1,de_x2,de_x3,x4,de_x5,de_x6]
        return res
    
if __name__ == '__main__':
    x = [torch.randn(16,64,32,32),torch.randn(16,128,16,16),torch.randn(16,512,8,8),torch.randn(16,1024,4,4),torch.randn(16,1024,2,2),torch.randn(16,1024,1,1)]
    conv_layer = Decoder(out_channels=1,generator_dim = 64 ,normalization='batchnorm')
    x = conv_layer(x)