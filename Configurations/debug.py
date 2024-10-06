
import os
dataPathRoot = '/data0/haochuan/'


hyperParams = {
        'seed':1,
        'debugMode':1,
        'expID':'Debug',# experiment name prefix
        'expDir': '/data-shared/server02/data1/haochuan/Character/DebugRecords202410/',
        
        # user interface
        'printInfoSeconds':3,

        # # devices
        # 'generatorDevice':'/device:CPU:0',
        # 'discrminatorDevice':'/device:CPU:0',
        # 'featureExtractorDevice':'/device:CPU:0',

        # # devices
        # 'device': 'GPU3',
        
        # data 
        # 'content_data_dir':# standard data location
        # [os.path.join(dataPathRoot, 'CASIA_Dataset/HandWritingData_OrgGrayScale/CASIA-HWDB1.1/')],

        # 'style_train_data_dir': # training data location
        # [os.path.join(dataPathRoot, 'CASIA_Dataset/HandWritingData_OrgGrayScale/CASIA-HWDB1.1/')],

        # 'style_validation_data_dir':# validation data location
        # [os.path.join(dataPathRoot, 'CASIA_Dataset/HandWritingData_OrgGrayScale/CASIA-HWDB1.1/')],

        'YamlPackage': '../YamlLists/Debug/',
        
        # 'contentYaml': # file list of the standard data
        # ['../YamlLists/Debug/Content.yaml'],

        # 'file_list_txt_style_train': # file list of the training data
        # ['../FileList/HandWritingData/Char_0_29_Writer_1001_1005_Isolated.txt'],

        # 'file_list_txt_style_validation': # file list of the validation data
        # ['../FileList/HandWritingData/Char_0_29_Writer_1001_1005_Isolated.txt'],
        
        'FullLabel0Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/Debug-Label0.txt',
        'FullLabel1Vec': '/data-shared/server09/data0/haochuan/CASIA_Dataset/LabelVecs/Debug-Label1.txt',

        
        # training configurations
        'trainAugmentContentTranslation':0,
        'trainAugmentContentRotation':0,
        'trainAugmentContentFlip':0,
        'trainAugmentStyleTranslation':0,
        'trainAugmentStyleRotation':0,
        'trainAugmentStyleFlip':0,
        'trainSplitContentStyleAugmentation':0,

        'testAugmentContentTranslation':0,
        'testAugmentContentRotation':0,
        'testAugmentContentFlip':0,
        'testAugmentStyleTranslation':0,
        'testAugmentStyleRotation':0,
        'testAugmentStyleFlip':0,
        'testSplitContentStyleAugmentation':0,
        
        
        'inputStyleNum':9, 
        'inputContentNum':5,

        # generator && discriminator
        #'generator': 'VanillaWNet-BasicBlockEncoder-AvgMaxResidualMixer-BasicBlockDecoder',
        # 'generator': 'TransWNet-AvgMaxConvResidualMixer-BasicBlockDecoder',
        # 'generator': 'PlainWNet-AvgMaxResidualMixer',
        # 'generator': 'SwinWNet-MixerMaxRes3@5-SwinBlockDecoder',
        # 'encoderArch': 'CV-CV-CV-ViT-CV',
        'discriminator':'NA',


        # input params
        'imgWidth':64,
        'channels':1,

        # optimizer setting
        'optimization_method':'adam',
        'initTrainEpochs':0,
        'final_learning_rate_pctg':0.01,

        # feature extractor parametrers
        'true_fake_target_extractor_dir': [],
        # 'content_prototype_extractor_dir':['/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG11Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG13Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG16Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/VGG19Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet18/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet34/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet50/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet101/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Content/ResNet152/BestExtractor.pth'],
        
        # 'style_reference_extractor_dir':  ['/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG11Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG13Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG16Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/VGG19Net/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet18/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet34/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet50/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet101/BestExtractor.pth',
        #                                    '/data-shared/server09/data1/haochuan/Codes/MuyinWNet/FeatureExtractorTrained/Debug/Ckpts/Style/ResNet152/BestExtractor.pth']
        
        'content_prototype_extractor_dir':[],
        'style_reference_extractor_dir':  []


}


penalties = {
        'generator_weight_decay_penalty': 0.0001,
        'discriminator_weight_decay_penalty':0.0003,
        'Pixel_Reconstruction_Penalty':1,
        'Lconst_content_Penalty':0.2,
        'Lconst_style_Penalty':0.2,
        'Discriminative_Penalty': 0,
        'Discriminator_Categorical_Penalty': 0,
        'Generator_Categorical_Penalty': 1,
        'Discriminator_Gradient_Penalty': 0,
        'Batch_StyleFeature_Discrimination_Penalty':0,
        # 'FeatureExtractorPenalty_ContentPrototype': [0.5,0.1,0.1,0.1,0.1,0.1],
        # 'FeatureExtractorPenalty_StyleReference':[1,0.5,0.5,0.3,0.3],
        # 'FeatureExtractorPenalty_ContentPrototype': [1,1,1,1,1,1,1,1,1],
        # 'FeatureExtractorPenalty_StyleReference':[1,1,1,1,1,1,1,1,1],
        
        'FeatureExtractorPenalty_ContentPrototype': [],
        'FeatureExtractorPenalty_StyleReference':[],
}

