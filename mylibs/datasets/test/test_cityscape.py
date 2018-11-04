# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, torch
from mylibs import Mask_RCNN_Dataset, CityscapeConfig, CityScapeDataset
from xinshuo_visualization import visualize_image
from xinshuo_io import save_image, mkdir_if_missing

dataset_dir = '/media/xinshuo/Data/Datasets/Cityscapes'
gttype='gtFine'
split = 'val'

config = CityscapeConfig()
dataset_cityscape = CityScapeDataset(dataset_dir, split=split, gttype=gttype)
dataset_cityscape.load_data()
dataset_cityscape.prepare()

masks, class_ids = dataset_cityscape.load_mask(0)
image = dataset_cityscape.load_image(0)

# Data generators
train_set = Mask_RCNN_Dataset(dataset_cityscape, config, augment=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

count = 1
vis_dir = os.path.join(dataset_dir, gttype, 'vis_%s' % split); mkdir_if_missing(vis_dir)
for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, image_index, filename in train_generator:
	print('CityScape Dataset Visualization Processing: %d' % count)
	dataset_cityscape.visualization(image_index.item(), save_dir=vis_dir)
	count += 1