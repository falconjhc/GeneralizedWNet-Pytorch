import sys
import torch
import torchvision.models as models
import torch.nn as nn
sys.path.append('./')


featureExtractors={
    'VGG11Net': models.vgg11_bn(),
    'VGG13Net': models.vgg13_bn(),
    'VGG16Net': models.vgg16_bn(),
    'VGG19Net': models.vgg19_bn(),
    'ResNet18': models.resnet18(),
    'ResNet34': models.resnet34(),
    'ResNet50': models.resnet50(),
    'ResNet101': models.resnet101(),
    'ResNet152': models.resnet152()
}


vggEvalLayers={
    'VGG11Net': [3, 6, 13, 20, 27],
    'VGG13Net': [5, 12, 19, 26, 33],
    'VGG16Net': [5, 12, 22, 32, 42],
    'VGG19Net': [5, 12, 25, 38, 51],
    
}

# from config import CONFIG as cfg
class FeatureExtractorBase(nn.Module):
    
    
    def __init__(self, outputNums, modelSelect, type):
        super(FeatureExtractorBase, self).__init__()
        if 'VGG' in modelSelect:
            self.extractor = VGGNets(output_nums=outputNums, modelSelect=modelSelect, type=type)
        elif 'Res' in modelSelect:
            self.extractor = RESNets(output_nums=outputNums, modelSelect=modelSelect, type=type)




class VGGNets(nn.Module):
    def __init__(self, output_nums, modelSelect, type):
        super(VGGNets, self).__init__()
        self.model = featureExtractors[modelSelect]
        self.evalLayer = vggEvalLayers[modelSelect]
        # self.model.avgpoolNew = nn.AdaptiveAvgPool2d((1, 1))
        self.model.classifierContent= nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Flatten(1,3),
            nn.Linear(4096, output_nums)
        )
        
        
        # self.model.classifierContent1=nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        # self.model.classifierContent2=nn.BatchNorm2d(1024)
        # self.model.classifierContent3=nn.ReLU(True)
        # self.model.classifierContent4=nn.Flatten(1,3)
        # self.model.classifierContent5=nn.Linear(4096, output_nums)
            
    
        
        
        
        self.model.classifierStyle= nn.Sequential(
            nn.Flatten(1,3),
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Linear(256, output_nums)
        )
        
        if type=='content':
            self.model.classifierNew = self.model.classifierContent
        elif type == 'style':
            self.model.classifierNew = self.model.classifierStyle

        
    def forward(self, x):
        intermediate_outputs = []
        # 将灰度图像的单通道复制到三个通道
        x = x.repeat(1, 3, 1, 1)
        # 注册钩子以获取每个特征层的输出,需要提取3，8，15，22，29
        for idx,layer in enumerate(self.model.features):
            x = layer(x)
            # print(idx, x.shape)
            if idx in self.evalLayer:
                intermediate_outputs.append(x)
        # 继续通过分类器部分
        # x = self.model.avgpoolNew(x)
        # x = torch.flatten(x, 1)
        x = self.model.classifierNew(x)
        # x = self.model.classifierContent2(x)
        # x = self.model.classifierContent3(x)
        # x = self.model.classifierContent4(x)
        # x = self.model.classifierContent5(x)
        return x,intermediate_outputs

    
class RESNets(nn.Module):
    def __init__(self, output_nums, modelSelect, type):
        super(RESNets, self).__init__()
        # models.vgg16().classifier
        self.model = featureExtractors[modelSelect]
        
        if '18' in modelSelect or '34' in modelSelect :
            self.expansion=1
        elif '101' in modelSelect or '152' in modelSelect or '50' in modelSelect:
            self.expansion=4
        
        
        #self.evalLayer = evalLayers[modelSelect]
        #self.extractor.classifier[-1] = nn.Linear(4096,output_nums)
        self.model.classifier = nn.Linear(512 * self.expansion, output_nums)
        
        
    def forward(self, x):
        intermediate_outputs = []
        # 将灰度图像的单通道复制到三个通道
        x = x.repeat(1, 3, 1, 1)
        
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        intermediate_outputs.append(x)
        x = self.model.maxpool(x)
        

        x = self.model.layer1(x)
        intermediate_outputs.append(x)
        
        x = self.model.layer2(x)
        intermediate_outputs.append(x)
        
        x = self.model.layer3(x)
        intermediate_outputs.append(x)
        
        x = self.model.layer4(x)
        intermediate_outputs.append(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.model.classifier(x)

        return x,intermediate_outputs