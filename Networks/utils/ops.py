import os
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
def create_if_not(path):
    #create path if not exist
    if not os.path.exists(path):
        os.makedirs(path)
    
class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=2, normalization=None):
        super(Convblock, self).__init__()
        ops = []

        ops.append(nn.LeakyReLU(negative_slope=0.2))
        # 定义卷积层
        ops.append(nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding))



        # 添加 Leaky ReLU 层
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=out_channels))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(out_channels))
        elif normalization != 'none':
            assert False

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class Deconvblock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=2, 
                 padding=2, normalization=None,dropout=True):
        super(Deconvblock, self).__init__()


        ops = []
        ops.append(nn.ReLU(inplace=True))

        # Perform transposed convolution
        ops.append(nn.ConvTranspose2d(in_channels, out_channels,kernel_size=kernel_size, stride=stride, padding=padding,output_padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(out_channels))
        elif normalization != 'none':
            assert False

        if dropout:
            ops.append(nn.Dropout2d(p=0.5, inplace=False))
    
        self.deconv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.deconv(x)
        return x
    
class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages,in_channels, out_channels,kernel_size=3, stride=1,padding='same',mixer='ResidualMixer',normalization=None):
        super(ResidualConvBlock, self).__init__()
        if padding.lower() == 'same':
            padding = kernel_size  // 2
        else:
            padding = 0

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = in_channels
            else:
                input_channel = out_channels

            ops.append(nn.Conv2d(input_channel, out_channels, kernel_size,stride, padding))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm2d(out_channels))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm2d(out_channels))
            else:
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))
                
        self.conv = nn.Sequential(*ops)


    def forward(self, x):
        x = (self.conv(x) + x)
        # x = self.relu(x)
        return x
    

def cv2torch(file_path,transform):
    return transform(Image.open(file_path).convert('L'))

def string2tensor(string):
    return torch.tensor(int(string))

def set_random(seed_id=1234):
    #set random seed for reproduce
    random.seed(seed_id)
    np.random.seed(seed_id)
    torch.manual_seed(seed_id)   #for cpu
    torch.cuda.manual_seed_all(seed_id) #for GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def read_file_to_dict(file_path):
    line_dict = {}
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file):
            line = int(line)
            line = str(line)
            # 移除每行末尾的换行符并存储到字典
            line_dict[line] = line_number
    return line_dict

def unormalize(tensor):
    # 反归一化操作
    tensor = tensor * 0.5 + 0.5  # 将 [-1, 1] 范围的值映射回 [0, 1]
    # 转换为 numpy 数组并且确保类型为 uint8
    output = tensor.int()
    return output

if __name__ == '__main__':
    # x = torch.randn(16,64,32,32)
    # in_channels = 64
    # out_channels = 128
    # conv_layer = Convblock(in_channels, out_channels, kernel_size=5, stride=2, padding='same', normalization='batchnorm')
    # x = conv_layer(x)
    # print(x.shape)
    
    # x = torch.randn(16,1024,1,1)
    # in_channels = 1024
    # out_channels = 512
    # deconv_layer = Deconvblock(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=2, padding='same', normalization='batchnorm',dropout=True)
    # x = deconv_layer(x)
    # print(x.shape)

    x = torch.randn(16,64,32,32)
    in_channels = 64
    out_channels = 64
    conv_layer = ResidualConvBlock(5,in_channels, out_channels, kernel_size=3, stride=1, padding='same', normalization='batchnorm')
    x = conv_layer(x)
    print(x.shape)