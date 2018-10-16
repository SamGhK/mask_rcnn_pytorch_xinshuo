# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, torch
from cityscape import CityscapeConfig, CityScapeDataset
from mylibs import Mask_RCNN_Dataset
from xinshuo_visualization import visualize_image
from xinshuo_io import save_image

dataset_dir = '/media/xinshuo/Data/Datasets/Cityscapes'
# year = '2014'
# download = False

config = CityscapeConfig()
dataset_cityscape = CityScapeDataset(dataset_dir, split='val', gttype='gtFine')
# dataset_cityscape.load_data(class_ids=[11, 12, 13])
# dataset_cityscape.load_data(class_ids=[11])
dataset_cityscape.load_data()
dataset_cityscape.prepare()

dataset_cityscape.visualization()
masks, class_ids = dataset_cityscape.load_mask(0)
print(masks.shape)
print(masks.dtype)
print(class_ids.shape)
print(class_ids.dtype)
print(class_ids)

image = dataset_cityscape.load_image(0)
save_image(image, save_path='img.jpg')

visualize_image(masks[:, :, 1], save_path='test.jpg')

# print(dataset_cityscape.image_ids)
print(dataset_cityscape.num_classes)
print(dataset_cityscape.source_class_ids)
# zxc

# Data generators
train_set = Mask_RCNN_Dataset(dataset_cityscape, config, augment=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)

count = 0
for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks in train_generator:
	# print()
	print(count)
	# print(images)

	count += 1