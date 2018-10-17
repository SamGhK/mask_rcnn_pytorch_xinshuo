# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, torch
from cityscape import CityscapeConfig, CityScapeDataset
from mylibs import Mask_RCNN_Dataset
from xinshuo_visualization import visualize_image
from xinshuo_io import save_image, mkdir_if_missing

dataset_dir = '/media/xinshuo/Data/Datasets/Cityscapes'
# year = '2014'
# download = False

config = CityscapeConfig()
dataset_cityscape = CityScapeDataset(dataset_dir, split='val', gttype='gtFine')
# dataset_cityscape.load_data(class_ids=[11, 12, 13])
# dataset_cityscape.load_data(class_ids=[11])
dataset_cityscape.load_data()
dataset_cityscape.prepare()


masks, class_ids = dataset_cityscape.load_mask(0)
# print(masks.shape)
# print(masks.dtype)
# print(class_ids.shape)
# print(class_ids.dtype)
# print(class_ids)

image = dataset_cityscape.load_image(0)
# save_image(image, save_path='img.jpg')


# visualize_image(masks[:, :, 1], save_path='test.jpg')

# print(dataset_cityscape.image_ids)
# print(dataset_cityscape.num_classes)
# print(dataset_cityscape.source_class_ids)
# zxc

# Data generators
train_set = Mask_RCNN_Dataset(dataset_cityscape, config, augment=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

count = 1
vis_dir = '/media/xinshuo/Data/Datasets/Cityscapes/gtFine/vis_val'; mkdir_if_missing(vis_dir)
for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, image_index, filename in train_generator:
	# print()
	print('processing %d' % count)
	# print(images)
	# print(image_index.item())
	# print(filename[0])
	dataset_cityscape.visualization(image_index.item(), save_dir=vis_dir)
	count += 1