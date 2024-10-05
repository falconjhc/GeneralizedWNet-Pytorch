import os
import yaml    
data_yaml = 'cmy/list/test.yaml'

if __name__ == '__main__':
    with open(data_yaml, 'r') as file:
        img_dict = yaml.safe_load(file)
    print(img_dict[0])
