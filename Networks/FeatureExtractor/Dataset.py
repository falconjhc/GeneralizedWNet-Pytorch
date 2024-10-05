import os
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml
sys.path.append('./')
from utils.ops import set_random,cv2torch,string2tensor,read_file_to_dict
from tqdm import tqdm
from torchvision.transforms import InterpolationMode


class FeatureExtractorDataset(Dataset):
    def __init__(self,config,is_train, augmentation):
        self.type = config.type
        self.augmentation = augmentation
        if config.debug:
            self.content_yaml = os.path.join(config.yamls, 'Content.yaml')
            self.train_style_yaml = os.path.join(config.yamls, 'TrainStyleReference.yaml')
            self.test_style_yaml = os.path.join(config.yamls, 'TestStyleReference.yaml')
        else:
            self.content_yaml = os.path.join(config.yamls, 'Content-PF64/Content.yaml')
            self.train_style_yaml = os.path.join(config.yamls, 'Style-PF80/TrainStyleReference.yaml')
            self.test_style_yaml = os.path.join(config.yamls, 'Style-PF80/TestStyleReference.yaml')
            
        # self.content_yaml = os.path.join(config.yamls, 'Content.yaml')
        # self.train_style_yaml = os.path.join(config.yamls, 'TrainStyleReference.yaml')
        # self.test_style_yaml = os.path.join(config.yamls, 'TestStyleReference.yaml')
        self.is_train = is_train
        self.train_set = []
        #self.num = config.outputnum
        
        self.order = read_file_to_dict(config.labelVecTxt)
        
        
        if self.type == 'content':
            with open(self.content_yaml, 'r', encoding='utf-8') as f:
                print("Loading "+ self.content_yaml + '...', end='\r')
                files = yaml.load(f.read(), Loader=yaml.FullLoader)
                print("Loading "+ self.content_yaml + ' completed.')
                
                for idx, (k,values) in tqdm(enumerate(files.items()),  total=len(files.items()), desc="Load the Train set"):
                    for file_path in values:
                        # file_path = os.path.join(data_path,value)
                        self.train_set.append((file_path ,self.order[k]))

        elif self.type == 'style':
            # files = {}
            # self.num = config.styleNum
            # order_txt = cfg['Label1_list']
            # data_path = os.path.join(base_dir,style_dir)
            # 读取参考样式的YAML文件
            with open(self.train_style_yaml, 'r', encoding='utf-8') as f:
                print("Loading "+ self.train_style_yaml + '...', end='\r')
                train_files = yaml.load(f.read(), Loader=yaml.FullLoader)
                # files.update(train_files)  # 将内容更新到files字典中
                print("Loading "+ self.train_style_yaml + ' completed.')

            # 读取验证样式的YAML文件
            with open(self.test_style_yaml, 'r', encoding='utf-8') as f:
                print("Loading "+ self.test_style_yaml + '...', end='\r')
                val_files = yaml.load(f.read(), Loader=yaml.FullLoader)
                # files.update(val_files)  # 将内容更新到files字典中
                print("Loading "+ self.test_style_yaml + ' completed.')

        

            for idx, (k,values) in tqdm(enumerate(train_files.items()),  total=len(train_files.items()), desc="Load the Train set"):
                for file_path in values:
                    # file_path = os.path.join(data_path,value)
                    self.train_set.append((file_path ,self.order[k]))
                    
            for idx, (k,values) in tqdm(enumerate(val_files.items()),  total=len(val_files.items()), desc="Load the Test set"):
                for file_path in values:
                    # file_path = os.path.join(data_path,value)
                    self.train_set.append((file_path ,self.order[k]))

        set_random()

    def __getitem__(self, index):
        if self.augmentation=='NA':
            transform = transforms.Compose([                
                transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
                transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
            ])
        elif self.augmentation == 'Half':
                transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(size=(64,64), scale=(0.75,1.0), antialias=True),                
                transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
                transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
            ])
        elif self.augmentation == 'Full':
                transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine(degrees=(-15,15), interpolation=InterpolationMode.BICUBIC, fill=(255), center=(32,32), translate=(0.15,0.15), scale=(0.75,1.25), shear=(15,15)),
                transforms.RandomResizedCrop(size=(64,64), scale=(0.75,1.0), antialias=True),                
                transforms.ToTensor(),  # 将 numpy.ndarray 转换为 torch.Tensor
                transforms.Normalize((0.5,), (0.5,)),  # 归一化到 [-1, 1]
                
            ])
        image = cv2torch(self.train_set[index][0],transform)
        label = string2tensor(self.train_set[index][1])

        raw_label = list(self.order.keys())[label]
        if not self.is_train:
            return image,label,raw_label
        else:
            return image,label
    
    def __len__(self):
        return len(self.train_set) 
    
    
# if __name__ == "__main__":
#     cfg['content_yaml'] = 'cmy/test_list/content_dir.yaml'
#     cfg['GT_style_yaml'] = 'cmy/test_list/train_GT_dir.yaml'
#     cfg['reference_style_yaml'] = 'cmy/test_list/train_reference_style_dir.yaml'
#     casia_dataset = Feature_Dataset(cfg,type='content',is_train=False)
#     casia_loader = DataLoader(casia_dataset, batch_size=8, shuffle=False,drop_last=True)
#     for image,label in casia_loader:
#         print(label)
