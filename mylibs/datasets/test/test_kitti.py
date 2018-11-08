# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, torch
from mylibs import Mask_RCNN_Dataset, KITTIConfig, KITTIDataset, unmold_image, kitti_class_names
from xinshuo_io import save_image, mkdir_if_missing, fileparts
from xinshuo_math import expand_mask
from xinshuo_visualization import visualize_image_with_bbox_mask, visualize_image
from xinshuo_visualization.python.private import save_vis_close_helper

dataset_dir = '/media/xinshuo/Data/Datasets/KITTI/semantics'
split = 'training'

config = KITTIConfig()
dataset_kitti = KITTIDataset(dataset_dir, split=split)
dataset_kitti.load_data()
dataset_kitti.prepare()

masks, class_ids = dataset_kitti.load_mask(0)
print(masks.dtype)
print(type(masks))
print(masks.shape)
print(class_ids)
image = dataset_kitti.load_image(0)
print(image.dtype)

# Data generators
train_set = Mask_RCNN_Dataset(dataset_kitti, config, augment=True)
train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

count = 1
vis_dir = os.path.join(dataset_dir, 'vis_%s' % split); mkdir_if_missing(vis_dir)
for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, image_index, filename in train_generator:
	print('KITTI Dataset Visualization Processing: %d' % count)
	# dataset_kitti.visualization(image_index.item(), save_dir=vis_dir)
	
	image_path = dataset_kitti.image_info[image_index.item()]['path']
	_, filename, _ = fileparts(image_path)

	images = images.numpy()[0].transpose(1, 2, 0)
	images = unmold_image(images, config)
	gt_masks, gt_boxes, gt_class_ids= gt_masks.numpy()[0], gt_boxes.numpy()[0].astype('int64'), gt_class_ids.numpy()[0]
	gt_masks = gt_masks.transpose(1, 2, 0)
	gt_masks = expand_mask(gt_boxes, gt_masks, images.shape)
	gt_boxes[:, [0, 1]] = gt_boxes[:, [1, 0]]
	gt_boxes[:, [2, 3]] = gt_boxes[:, [3, 2]]

	fig, _ = visualize_image_with_bbox_mask(images, boxes=gt_boxes, masks=gt_masks, class_ids=gt_class_ids, class_names=['BG'] + kitti_class_names)
	save_path_tmp = os.path.join(vis_dir, filename+'.jpg')
	save_vis_close_helper(fig=fig, transparent=False, save_path=save_path_tmp)

	count += 1