# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, torch, numpy as np, matplotlib.pyplot as plt
from config import Config
if "DISPLAY" not in os.environ: plt.switch_backend('agg')
from mylibs import MaskRCNN, class_names
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing, load_image, save_image
from xinshuo_visualization import visualize_image_with_bbox_mask
from xinshuo_visualization.python.private import save_vis_close_helper
from xinshuo_miscellaneous import convert_secs2time, Timer

dataset = 'coco'
KITTI_class_mapping = {0: 'Background', 1: 'Pedestrian', 2: 'Car', 3: 'Cyclist'}

def class_mapping_coco_to_kitti(class_id):
	if class_id in [1]:
		return 1		# pedestrian
	# elif class_id in [2]:
		# return 3		# cyclist
	elif class_id in [3]:
		return 2		# car
	else: return 0

def class_mapping_cityscape_to_kitti(class_id):
	if class_id in [1]:
		return 1		# pedestrian
	elif class_id in [2]:
		return 3		# cyclist
	elif class_id in [3]:
		return 2		# car
	else: return 0

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'evaluate_%s' % dataset
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # DETECTION_MIN_CONFIDENCE = 0
    if dataset == 'coco': NUM_CLASSES = 1 + 80
    else: NUM_CLASSES = 1 + len(class_names)

config = InferenceConfig()

root_dir = os.getcwd()                      # Root directory of the project
log_dir = os.path.join(root_dir, 'logs')    # Directory to save logs and trained model


# model_path = os.path.join(root_dir, 'resnet50_imagenet.pth')    # Path to trained weights file
model_path = os.path.join(root_dir, 'mask_rcnn_coco.pth')    # Path to trained weights file
# model_path = '/media/xinshuo/Data/models/mask_rcnn_pytorch/coco20181015T1656/mask_rcnn_coco_0160.pth'
# model_path = '/media/xinshuo/Data/models/mask_rcnn_pytorch/coco20181015T1653/mask_rcnn_coco_0160.pth'
# model_path = '/media/xinshuo/Data/models/mask_rcnn_pytorch/cityscape20181018T0035/mask_rcnn_cityscape_0060.pth'

# images_dir = os.path.join(root_dir, 'images')    # Directory of images to run detection on
# save_dir = os.path.join(root_dir, 'tmp/results'); mkdir_if_missing(save_dir)

data_dir = '/media/xinshuo/Data/Datasets/KITTI/object/training'
images_dir = os.path.join(data_dir, 'image_2')
save_dir = os.path.join(data_dir, 'results/mask_preprocessed'); mkdir_if_missing(save_dir)
label_bbox_match_dir = os.path.join(save_dir, 'label_bbox_matching'); mkdir_if_missing(label_bbox_match_dir)

# data_dir = '/media/xinshuo/Data/Datasets/Cityscapes/leftImg8bit'
# images_dir = os.path.join(data_dir, 'val/frankfurt')
# save_dir = os.path.join(data_dir, 'results/mask_preprocessed_cityscape'); mkdir_if_missing(save_dir)

vis_dir = os.path.join(save_dir, 'visualization'); mkdir_if_missing(vis_dir)
mask_dir = os.path.join(save_dir, 'masks'); mkdir_if_missing(mask_dir)
detection_result_filepath = os.path.join(save_dir, 'mask_results.txt'); detection_results_file = open(detection_result_filepath, 'w')

config = InferenceConfig()
# config.display()

# Create model object.
model = MaskRCNN(model_dir=log_dir, config=config)
if config.GPU_COUNT: model = model.cuda()
model.load_weights(model_path)    # Load weights trained on MS-COCO

# Index of the class in the list is its ID. For example, to get ID of # the teddy bear class, use: class_names.index('teddy bear')
if dataset == 'coco':
	class_names_bg = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
		'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
		'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
		'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
		'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
		'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
		'teddy bear', 'hair drier', 'toothbrush']
elif dataset == 'cityscape': class_names_bg = ['BG'] + class_names
else: assert False, 'error'

# load the data
image_list, num_list = load_list_from_folder(images_dir)
print('testing results on %d images' % num_list) 
count = 1
timer = Timer(); timer.tic()
# for index in range(938, num_list):
for image_file_tmp in image_list:
	# image_file_tmp = image_list[index]
	_, filename, _ = fileparts(image_file_tmp)
	label_matching_file_tmp	= os.path.join(label_bbox_match_dir, filename+'.txt')
	label_matching_file_tmp = open(label_matching_file_tmp, 'w')

	image = load_image(image_file_tmp)
	results = model.detect([image])		# inference, results is a dictionary
	if len(results) == 0: 
		count += 1
		print('testing %d/%d, no detected results' % (count, num_list))
		continue

	# visualize and save results
	r = results[0]			# results from the first image
	num_instances = r['masks'].shape[-1]
	# fig, _ = visualize_image_with_bbox_mask(image, boxes=r['rois'], masks=r['masks'], class_ids=r['class_ids'], class_names=class_names_bg, scores=r['scores'])
	save_path_tmp = os.path.join(vis_dir, filename+'.jpg')
	# save_vis_close_helper(fig=fig, transparent=False, save_path=save_path_tmp)

	elapsed = timer.toc(average=False)
	remaining_str = convert_secs2time((elapsed) / count * (num_list - count))
	elapsed_str = convert_secs2time(elapsed)
	print('testing %d/%d, detected %d instances, EP: %s, ETA: %s, saving to %s' % (count, num_list, num_instances, elapsed_str, remaining_str, save_path_tmp))

	# save data for each individual instances
	for instance_index in range(num_instances):
		mask_tmp = r['masks'][:, :, instance_index]
		class_tmp = r['class_ids'][instance_index]
		bbox_tmp = r['rois'][instance_index, :]		# y1, x1, y2, x2

		if dataset == 'coco': class_tmp = class_mapping_coco_to_kitti(class_tmp)
		elif dataset == 'cityscape': class_tmp = class_mapping_cityscape_to_kitti(class_tmp)
		else: assert False, 'error'
		if class_tmp == 0: continue

		score_tmp = r['scores'][instance_index]
		mask_tmp *= 255
		mask_dir_frame = os.path.join(mask_dir, filename); mkdir_if_missing(mask_dir_frame)
		save_path_tmp = os.path.join(mask_dir_frame, 'instance_%04d'%instance_index+'.jpg')		
		# save_image(mask_tmp, save_path=save_path_tmp)
		
		# save info for every instances
		save_str = '%s %s %.2f %.2f %.2f %.2f %.2f %s \n' % (image_file_tmp, class_tmp, bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3], score_tmp, save_path_tmp)
		detection_results_file.write(save_str)

		label_matching_str = '%s %.2f %.2f %.2f %.2f %.2f %s\n' % (KITTI_class_mapping[class_tmp], bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3], score_tmp, save_path_tmp)
		label_matching_file_tmp.write(label_matching_str)

	count += 1
	label_matching_file_tmp.close()

detection_results_file.close()