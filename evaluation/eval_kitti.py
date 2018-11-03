# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, torch, numpy as np, matplotlib.pyplot as plt
if "DISPLAY" not in os.environ: plt.switch_backend('agg')
from mylibs import MaskRCNN, cityscape_class_names, Config, coco_class_names
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing, load_image, save_image
from xinshuo_visualization import visualize_image_with_bbox_mask
from xinshuo_visualization.python.private import save_vis_close_helper
from xinshuo_miscellaneous import convert_secs2time, Timer, get_timestring, print_log

train_dataset = 'coco'
object_interest = ['Pedestrian', 'Car', 'Cyclist']
data_dir = '/media/xinshuo/Data/Datasets/KITTI/object/training'
results_name = 'maskrcnn_bbox_detection_results_%s_%s' % (train_dataset, get_timestring())
root_dir = os.getcwd()                      # Root directory of the project

# Index of the class in the list is its ID. For example, to get ID of # the teddy bear class, use: class_names.index('teddy bear')
if train_dataset == 'coco': class_names_bg = coco_class_names
elif train_dataset == 'cityscape': class_names_bg = ['BG'] + class_names
else: assert False, 'error'

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'evaluate_%s' % train_dataset
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # DETECTION_MIN_CONFIDENCE = 0
    if train_dataset == 'coco': NUM_CLASSES = 1 + 80
    else: NUM_CLASSES = 1 + len(class_names)
config = InferenceConfig()

##--------------------------------- Data Directory ----------------------------------##
images_dir = os.path.join(data_dir, 'image_2')
save_dir = os.path.join(data_dir, 'results/%s' % results_name); mkdir_if_missing(save_dir)
vis_dir = os.path.join(save_dir, 'visualization'); mkdir_if_missing(vis_dir)
log_file = os.path.join(save_dir, 'log.txt'); log_file = open(log_file, 'w')
bbox_eval_folder = os.path.join(save_dir, 'data'); mkdir_if_missing(bbox_eval_folder)

##--------------------------------- Model Directory ----------------------------------##
# model_path = os.path.join(root_dir, 'resnet50_imagenet.pth')    		# Path to trained weights from Imagenet
model_path = os.path.join(root_dir, '../models/mask_rcnn_coco.pth')    			# Path to trained weights file
# model_path = '/media/xinshuo/Data/models/mask_rcnn_pytorch/cityscape20181018T0035/mask_rcnn_cityscape_0060.pth'
model = MaskRCNN(model_dir=save_dir, config=config)			# Create model object.
if config.GPU_COUNT: model = model.cuda()
model.load_weights(model_path)    # Load weights 

##--------------------------------- Build KITTI Evaluation Results ----------------------------------##
image_list, num_list = load_list_from_folder(images_dir)
print('KITTI Evaluation: testing results on %d images' % num_list) 
count = 1
timer = Timer(); timer.tic()
# for index in range(938, num_list):
for image_file_tmp in image_list:
	# image_file_tmp = image_list[index]
	_, filename, _ = fileparts(image_file_tmp)
	bbox_eval_file_tmp = os.path.join(bbox_eval_folder, filename+'.txt'); bbox_eval_file_tmp = open(bbox_eval_file_tmp, 'w')

	image = load_image(image_file_tmp)
	results = model.detect([image])		# inference, results is a dictionary
	if len(results) == 0: 
		count += 1
		print('testing %d/%d, no detected results' % (count, num_list))
		continue

	# visualize and save results
	r = results[0]			# results from the first image in the batch 
	num_instances = r['masks'].shape[-1]
	bboxes_tmp = r['rois']			# y1, x1, y2, x2 format
	bboxes_tmp[:, [0, 1]] = bboxes_tmp[:, [1, 0]]
	bboxes_tmp[:, [2, 3]] = bboxes_tmp[:, [3, 2]]			# TLBR format

	# visualization
	fig, _ = visualize_image_with_bbox_mask(image, boxes=bboxes_tmp, masks=r['masks'], class_ids=r['class_ids'], class_names=class_names_bg, scores=r['scores'])
	save_path_tmp = os.path.join(vis_dir, filename+'.jpg')
	save_vis_close_helper(fig=fig, transparent=False, save_path=save_path_tmp)

	elapsed = timer.toc(average=False)
	remaining_str = convert_secs2time((elapsed) / count * (num_list - count))
	elapsed_str = convert_secs2time(elapsed)
	print_str = 'KITTI Evaluation: testing %d/%d, detected %d instances, EP: %s, ETA: %s' % (count, num_list, num_instances, elapsed_str, remaining_str)
	print(print_str)
	print_log('%s, saving to %s' % (print_str, filename), log=log_file, display=False)

	# save data for each individual instances
	for instance_index in range(num_instances):
		class_tmp = r['class_ids'][instance_index]
		if train_dataset == 'coco': class_tmp = class_names_bg[class_tmp]
		elif train_dataset == 'cityscape': class_tmp = class_names_bg[class_tmp]
		else: assert False, 'error'
		if not (class_tmp in object_interest): continue

		bbox_tmp = bboxes_tmp[instance_index, :]		# TLBR format
		score_tmp = r['scores'][instance_index]
		
		# save info for every instances
		save_str = '%s -1 -1 -10 %.2f %.2f %.2f %.2f 1 1 1 0 0 0 0 %.4f \n' % (class_tmp, bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3], score_tmp)
		bbox_eval_file_tmp.write(save_dir)
		
	count += 1
	bbox_eval_file_tmp.close()
