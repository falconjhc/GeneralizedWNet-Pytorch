CONFIG = {  
            # # trainset
            # 'content_yaml':'cmy/CASIA/content_dir.yaml',
            # 'GT_style_yaml':'cmy/CASIA/train_GT_dir.yaml',
            # 'reference_style_yaml':'cmy/CASIA/train_reference_style_dir.yaml',
            # # 'Label0_list': 'cmy/CASIA/PF80-Label0.txt',
            # # 'Label1_list': 'cmy/CASIA/PF80-Label1.txt',
            
            # # val
            # 'val_GT_style_yaml':'cmy/CASIA/val_GT_dir.yaml',
            # 'val_reference_style_yaml':'cmy/CASIA/val_reference_style_dir.yaml',

            # # 'inputStyleNum':5,
            # # 'inputContentNum':64,
            # 'in_channels':1,
            # 'generator_dim':64,
            
            # FeatureExtractor
            # 'content_extractor_ckpt':'/data1/muyinchen/WNet/Wnet_fullcontent/Networks/FeatureExtractor/ckpt/content_best_extractor_model.pth',
            # 'style_extractor_ckpt':'/data1/muyinchen/WNet/Wnet_fullcontent/Networks/FeatureExtractor/ckpt/style_best_extractor_model.pth',

            'batch_size':64,
            'num_workers':16,
            'max_epoch':500,
            'lr':0.01,

            'generator_weight_decay':0.0001,  
            'Pixel_Reconstruction_Penalty':1,
            'Lconst_content_Penalty':0.2,
            'Lconst_style_Penalty':0.2,
            'Generator_Categorical_Penalty':1,
            'FeatureExtractorPenalty_ContentPrototype':0.5,
            'FeatureExtractorPenalty_StyleReference':1,

            }


