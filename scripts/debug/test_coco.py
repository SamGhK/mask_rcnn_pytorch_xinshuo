# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, numpy as np, torch
from coco import CocoConfig, CocoDataset, evaluate_coco
from mylibs import Mask_RCNN_Dataset
from xinshuo_visualization import visualize_image
from xinshuo_io import save_image

dataset_dir = '/media/xinshuo/Data/Datasets/coco'
year = '2014'
download = False

config = CocoConfig()
dataset_coco = CocoDataset()
dataset_coco.load_data(dataset_dir, 'val', year=year, auto_download=False)
dataset_coco.prepare()
# print(dataset_coco.image_info[3]['id'])
masks, class_ids = dataset_coco.load_mask(0)
print(masks.shape)
print(masks.dtype)
print(class_ids.shape)
print(class_ids.dtype)
print(class_ids)

img = dataset_coco.load_image(0)
save_image(img, save_path='img.jpg')

visualize_image(masks[:, :, -1], save_path='test.jpg')

image = dataset_coco.load_image(0)

# print(dataset_coco.image_ids)
print(dataset_coco.num_classes)
print(dataset_coco.source_class_ids)
# zxc

# Data generators
train_set = Mask_RCNN_Dataset(dataset_coco, config, augment=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)

count = 0
for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks in train_generator:
	# print()
	# print(count)
	# print(images)

	count += 1