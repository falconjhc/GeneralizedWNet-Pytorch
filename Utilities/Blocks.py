import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import torch
import torch.nn.functional as F
import numpy as np 
# from Utilities.utils import FindKeys
import torch.nn as nn


mlpRatio=4
patchSize=4


import torch
import torch.nn.functional as F
from Utilities.VisionTransformer import VisionTransformer as VIT
from Utilities.VisionTransformer import PatchMerging as PatchMerger
from Utilities.VisionTransformer import PatchExpansion as PatchExpander

from torchvision.models.resnet import  conv1x1, conv3x3
import torch.nn as nn

def deconv1x1(in_planes, out_planes, stride=1):
    """1x1 transposed convolution (deconvolution) with stride and output padding for exact size"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, 
                              output_padding=stride-1, bias=False)
def deconv3x3(in_planes, out_planes, stride=1):
    """3x3 transposed convolution (deconvolution) to double the spatial size"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, 
                              output_padding=stride-1, bias=False)
    
    
class BlockFeature(object):
    def __init__(self, cnn, vit=None):
        self.cnn = cnn
        if vit is not None:
            self.vit = vit
        elif cnn.shape[2] >= patchSize:  # In PyTorch, height is at index 2 for (batch, channels, height, width)
            # Use unfold to extract patches in PyTorch
            vit = cnn.unfold(2, patchSize, patchSize).unfold(3, patchSize, patchSize)
            
            # Extract patchW and patchH based on the unfolded dimensions
            patchW = vit.size(2)  # Equivalent to vit.shape[1] in TensorFlow
            patchH = vit.size(3)  # Equivalent to vit.shape[2] in TensorFlow
            
            # Reshape vit to match the expected shape
            self.vit = vit.contiguous().view(vit.size(0), patchW * patchH, patchSize * patchSize * cnn.size(1))
        else:
            self.vit = None
    def ProcessOutputToList(self):
        if self.vit is not None:
            return [str(self.cnn.shape)[11:-1], str(self.vit.shape)[11:-1]]
        else:
            return [str(self.cnn.shape)[11:-1], 'None']
            

class PatchExtractor(nn.Module):
    def __init__(self, featureDim, patchSize=4):
        super(PatchExtractor, self).__init__()
        self.patchSize=patchSize
        self.featureDim=featureDim
        self.patchDim = self.patchSize*self.patchSize*self.featureDim
        
    def forward(self, x):
        batch_size = x.size(0)  # Get batch size from images tensor
        patches = x.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize)
        patches = patches.contiguous().view(batch_size, -1, self.patchSize * self.patchSize * x.size(1))  # Reshape the patches        
        return patches
    
class EncodingBottleneckBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(EncodingBottleneckBlock,self).__init__()
        if inDims['HW']==outDims['HW']*2 or config['downsample']:
            stride=2
        else:
            stride=1
        self.conv_identity = conv1x1(inDims['MapC'], outDims['MapC'], stride)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
        
        self.conv1 = conv1x1(inDims['MapC'], outDims['MapC'])
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        self.conv2 = conv3x3(outDims['MapC'], outDims['MapC'], stride)
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        self.conv3 = conv1x1(outDims['MapC'], outDims['MapC'])
        self.bn3 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = nn.ReLU(inplace=True)
        # self.downsample = downsample
        self.stride = stride
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
            
        
    def forward(self, x):
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)

        outcnn = self.conv1(x.cnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)

        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)
        outcnn = self.relu(outcnn)

        outcnn = self.conv3(outcnn)
        outcnn = self.bn3(outcnn)
        
        outcnn += identity
        outcnn = self.relu(outcnn)
        
        outvit = self.patchExtractor(outcnn)
        out  =BlockFeature(cnn=outcnn, vit=outvit)
        return out
    
class EncodingBasicBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(EncodingBasicBlock,self).__init__()
        if inDims['HW']==outDims['HW']*2 or config['downsample']:
            stride=2
        else:
            stride=1
            
        self.conv_identity = conv3x3(inDims['MapC'], outDims['MapC'], stride)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
            
        self.conv1 = conv3x3(inDims['MapC'], outDims['MapC'], stride)
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outDims['MapC'], outDims['MapC'])
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        self.stride = stride
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
    
    def forward(self,x):
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)

        outcnn = self.conv1(x.cnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)

        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)

        outcnn += identity
        outcnn = self.relu(outcnn)
        outvit = self.patchExtractor(outcnn)
        out =BlockFeature(cnn=outcnn, vit=outvit)
        return out
    

class EncodingVisionTransformerBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(EncodingVisionTransformerBlock,self).__init__()
        
        
        self.downsample = False
        self.inDims=inDims
        self.outDims=outDims
        if inDims['VitDim'] == outDims['VitDim'] * 4 or config['downsample']:
            self.downsample = True

        _, numVit, numHead = config['option'].split("@")
        numVit = int(numVit)
        numHead = int(numHead)
        
        self.vit = VIT(image_size=inDims['HW'],
                       patch_size=patchSize,
                       num_layers=numVit, num_heads=numHead,
                       d_model=outDims['VitC'],
                       mlp_dim=outDims['VitDim']*mlpRatio,
                       patchDim = inDims['PatchDim'])
        self.merger = PatchMerger(dim=outDims['VitC'])
        
    def forward(self,x):
        outvit = self.vit(x.vit)
        if self.downsample:
            outvit=self.merger(outvit)
        
        count=1
        for ii in range(len(outvit.shape)-1): count = count * outvit.shape[ii+1]
        if count%(self.outDims['HW']*self.outDims['HW'])==0:
            xcnn = outvit.view(outvit.size(0), -1, self.outDims['HW'],self.outDims['HW'])
        else:
            xcnn=None
        return BlockFeature(cnn=xcnn, vit=outvit)
        
        
        
class DecodingVisionTransformerBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(DecodingVisionTransformerBlock,self).__init__()
        
        
        self.upsample = False
        self.inDims=inDims
        self.outDims=outDims
        if inDims['VitDim'] == outDims['VitDim'] // 4 or config['upsample']:
            self.upsample = True
            

        _, numVit, numHead = config['option'].split("@")
        numVit = int(numVit)
        numHead = int(numHead)
        
        self.vit = VIT(image_size=outDims['HW'],
                       patch_size=patchSize,
                       num_layers=numVit, num_heads=numHead,
                       d_model=outDims['VitC'],
                       mlp_dim=outDims['VitDim']*mlpRatio,
                       patchDim = inDims['PatchDim'])
        self.expander = PatchExpander(dim=outDims['VitDim'])
        
    def forward(self,x, enc=None):
        if enc is not None:
            outvit = torch.concat((x.vit, enc.vit), dim=-1)
        else:
            outvit = x.vit 
            
        if self.upsample:
            outvit = self.expander(outvit)
        else:
            outvit=x.vit
        
        
        
        outvit = self.vit(outvit)
            
        # outvit = self.vit(x.vit)
        # if self.downsample:
        #     outvit=self.merger(outvit)
        xcnn = outvit.view(outvit.size(0), -1, self.outDims['HW'],self.outDims['HW'])
        return BlockFeature(cnn=xcnn, vit=outvit)
    
    
        
class DecodingBasicBlock(nn.Module):
    def __init__(self, inDims, outDims, config):
        super(DecodingBasicBlock,self).__init__()
        self.lastLayer = config['lastLayer']
        if inDims['HW']==outDims['HW']//2 or config['upsample']:
            stride=2
        else:
            stride=1
        self.conv_identity = deconv3x3(inDims['MapC'], outDims['MapC'], stride)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
        
        
        if config['skip']  is not None:
            self.conv1 = deconv3x3(inDims['MapC']+config['skip'][0], outDims['MapC'], stride)
        else:
            self.conv1 = deconv3x3(inDims['MapC'], outDims['MapC'], stride)
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.conv2 = deconv3x3(outDims['MapC'], outDims['MapC'])
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        self.stride = stride
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
    
    def forward(self, x, enc=None):
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)
        if enc is not None:
            outcnn = torch.concat((x.cnn, enc.cnn), dim=1)
        else:
            outcnn=x.cnn
        outcnn = self.conv1(outcnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)
        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)

        outcnn += identity
        if not self.lastLayer:
            outcnn = self.relu(outcnn)
        else:
            outcnn = self.tanh(outcnn)
        outvit = self.patchExtractor(outcnn)
        out =BlockFeature(cnn=outcnn, vit=outvit)
        return out


class DecodingBottleneckBlock(nn.Module):
    def __init__(self, inDims, outDims,  config):
        super(DecodingBottleneckBlock,self).__init__()
        self.lastLayer = config['lastLayer']
        if inDims['HW']==outDims['HW']//2 or config['upsample']:
            stride=2
        else:
            stride=1
        self.conv_identity = deconv1x1(inDims['MapC'], outDims['MapC'], stride)
        self.bn_identity = nn.BatchNorm2d(outDims['MapC'])
        
        self.conv1 = deconv1x1(inDims['MapC'], outDims['MapC'])
        self.bn1 = nn.BatchNorm2d(outDims['MapC'])
        
        
        
        if config['skip'] is not None:
            self.conv2 = deconv3x3(outDims['MapC']+config['skip'][0], outDims['MapC'], stride)
        else:
            self.conv2 = deconv3x3(outDims['MapC'], outDims['MapC'], stride)
        
        
        # self.conv2 = deconv3x3(outDims['MapC'], outDims['MapC'], stride)
        self.bn2 = nn.BatchNorm2d(outDims['MapC'])
        
        self.conv3 = deconv1x1(outDims['MapC'], outDims['MapC'])
        self.bn3 = nn.BatchNorm2d(outDims['MapC'])
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        # self.downsample = downsample
        self.stride = stride
        self.patchExtractor=PatchExtractor(featureDim=outDims['MapC'])
            
        
    def forward(self, x, enc=None):
        identity = self.conv_identity(x.cnn)
        identity = self.bn_identity(identity)

        outcnn = self.conv1(x.cnn)
        outcnn = self.bn1(outcnn)
        outcnn = self.relu(outcnn)

        
        if enc is not None:
            outcnn = torch.concat((outcnn, enc.cnn), dim=1)
            
        outcnn = self.conv2(outcnn)
        outcnn = self.bn2(outcnn)
        outcnn = self.relu(outcnn)

        outcnn = self.conv3(outcnn)
        outcnn = self.bn3(outcnn)
        
        outcnn += identity
        if not self.lastLayer:
            outcnn = self.relu(outcnn)
        else:
            outcnn = self.tanh(outcnn)
        outvit = self.patchExtractor(outcnn)
        out  =BlockFeature(cnn=outcnn, vit=outvit)
        return out