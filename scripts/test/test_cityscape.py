# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os
from cityscape import CityscapeConfig, CityScapeDataset

dataset_dir = '/media/xinshuo/Data/Datasets/Cityscape'
# year = '2014'
# download = False

dataset_coco = CityScapeDataset()
dataset_coco.load_data(dataset_dir, 'val')
dataset_coco.prepare()