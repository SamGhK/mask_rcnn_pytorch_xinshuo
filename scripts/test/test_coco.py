# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, numpy as np
from coco import CocoConfig, CocoDataset, evaluate_coco
from xinshuo_visualization import visualize_image
from xinshuo_io import save_image

dataset_dir = '/media/xinshuo/Data/Datasets/coco'
year = '2014'
download = False


dataset_coco = CocoDataset()
dataset_coco.load_data(dataset_dir, 'val', year=year, auto_download=False)
dataset_coco.prepare()
# print(dataset_coco.image_info[3]['id'])
mask, class_ids = dataset_coco.load_mask(0)
print(mask)
print(mask.shape)
print(np.max(mask[:, :, :]))
print(np.min(mask[:, :, :]))
print(mask.dtype)



img = dataset_coco.load_image(0)
save_image(img, save_path='img.jpg')

visualize_image(mask[:, :, -1], save_path='test.jpg')
print(class_ids)