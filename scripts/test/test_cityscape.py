# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os
from cityscape import CityscapeConfig, CityScapeDataset

dataset_dir = '/media/xinshuo/Data/Datasets/Cityscapes'
# year = '2014'
# download = False

dataset_cityscape = CityScapeDataset(dataset_dir, 'val')
# dataset_cityscape.load_data(class_ids=[11, 12, 13])
dataset_cityscape.load_data(class_ids=[11])
# dataset_cityscape.load_data()
dataset_cityscape.prepare()
dataset_cityscape.visualization()