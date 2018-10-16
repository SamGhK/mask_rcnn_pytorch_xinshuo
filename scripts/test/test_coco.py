# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os
from coco import CocoConfig, CocoDataset, evaluate_coco

dataset_dir = '/media/xinshuo/Data/Datasets/coco'
year = '2014'
download = False


dataset_coco = CocoDataset()
dataset_coco.load_data(dataset_dir, 'val', year=year, auto_download=False)
dataset_coco.prepare()