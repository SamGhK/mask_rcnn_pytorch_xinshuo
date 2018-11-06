# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, torch
from mylibs import Mask_RCNN_Dataset, KITTIConfig, KITTIDataset
from xinshuo_visualization import visualize_image
from xinshuo_io import save_image, mkdir_if_missing

dataset_dir = '/media/xinshuo/Data/Datasets/KITTI/semantics'
split = 'training'

config = KITTIConfig()
dataset_kitti = KITTIDataset(dataset_dir, split=split)
dataset_kitti.load_data()
dataset_kitti.prepare()

masks, class_ids = dataset_kitti.load_mask(0)
image = dataset_kitti.load_image(0)

# Data generators
train_set = Mask_RCNN_Dataset(dataset_kitti, config, augment=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

count = 1
vis_dir = os.path.join(dataset_dir, 'vis_%s' % split); mkdir_if_missing(vis_dir)
for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, image_index, filename in train_generator:
	print('KITTI Dataset Visualization Processing: %d' % count)
	dataset_kitti.visualization(image_index.item(), save_dir=vis_dir)
	count += 1