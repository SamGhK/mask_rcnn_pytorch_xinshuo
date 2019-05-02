# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com
import os, torch, numpy as np, matplotlib.pyplot as plt
if "DISPLAY" not in os.environ: plt.switch_backend('agg')
from mylibs import MaskRCNN, coco_class_names, Config, cityscape_class_names
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing, load_image, save_image
from xinshuo_visualization import visualize_image_with_bbox_mask, visualize_image
from xinshuo_visualization.python.private import save_vis_close_helper
from xinshuo_miscellaneous import convert_secs2time, Timer, print_log

train_dataset = 'coco'
root_dir = os.getcwd()                      # Root directory of the project

# Index of the class in the list is its ID. For example, to get ID of # the teddy bear class, use: class_names.index('teddy bear')
if train_dataset == 'coco': class_names_bg = coco_class_names
elif train_dataset == 'cityscape': class_names_bg = ['BG'] + cityscape_class_names
else: assert False, 'error'

class InferenceConfig(Config):
    # Set batch size to 1 since we'll be running inference on one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    NAME = 'demo_%s' % train_dataset
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0
    if train_dataset == 'coco': NUM_CLASSES = 1 + 80
    else: NUM_CLASSES = 1 + len(class_names)
config = InferenceConfig()

##--------------------------------- Data Directory ----------------------------------##
# KITTI
# data_dir = '/media/xinshuo/Data/Datasets/KITTI/object/training'
# images_dir = os.path.join(data_dir, 'image_2')
# save_dir = os.path.join(data_dir, 'results/mask_preprocessed'); mkdir_if_missing(save_dir)

# Cityscape
# data_dir = '/media/xinshuo/Data/Datasets/Cityscapes/leftImg8bit'
# images_dir = os.path.join(data_dir, 'val/frankfurt')
# save_dir = os.path.join(data_dir, 'results/mask_preprocessed_cityscape'); mkdir_if_missing(save_dir)

# Shimizu
# folder_name = 'Q_C0006_330-360sec'
# folder_name = 'Q_C0006_480-510sec'
data_dir = '/media/xinshuo/Data/Datasets/shimizu'
# data_dir = '/media/xinshuo/Data/Datasets/Cityscapes/leftImg8bit/demoVideo'
images_dir = os.path.join(data_dir, 'images_every10')
# images_dir = os.path.join(data_dir, 'stuttgart_00')
# images_dir = os.path.join(data_dir, 'public_asset')
save_dir = os.path.join(data_dir, 'results', 'MASK_RCNN'); mkdir_if_missing(save_dir)
# save_dir = os.path.join(data_dir, 'results', 'public_asset'); mkdir_if_missing(save_dir)

# vis_dir = os.path.join(data_dir, 'visualization', 'MASK_RCNN'); mkdir_if_missing(vis_dir)
# vis_dir = os.path.join(data_dir, 'visualization', 'public_asset'); mkdir_if_missing(vis_dir)
vis_dir = os.path.join(save_dir, 'visualization'); mkdir_if_missing(vis_dir)
mask_dir = os.path.join(save_dir, 'masks'); mkdir_if_missing(mask_dir)
detection_result_filepath = os.path.join(save_dir, 'mask_results.txt'); detection_results_file = open(detection_result_filepath, 'w')
log_file = os.path.join(save_dir, 'log.txt'); log_file = open(log_file, 'w')

##--------------------------------- Model Directory ----------------------------------##
# model_path = os.path.join(root_dir, 'resnet50_imagenet.pth')    # Path to trained weights file
if train_dataset == 'coco': model_path = os.path.join(root_dir, 'models/mask_rcnn_coco.pth')    # Path to trained weights file
else: assert False, 'error'

model = MaskRCNN(model_dir=save_dir, config=config)		# Create model object.
if config.GPU_COUNT: model = model.cuda()
model.load_weights(model_path)    # Load weights 

##--------------------------------- Testing ----------------------------------##
image_list, num_list = load_list_from_folder(images_dir, ext_filter=['.png', '.jpg'], depth=2)
print_log('testing results on %d images' % num_list, log=log_file) 
count = 1
timer = Timer(); timer.tic()
for image_file_tmp in image_list:
	parent_dir, filename, _ = fileparts(image_file_tmp)
	video_dir = parent_dir.split('/')[-1]
	# print(video_dir)
	# zxc

	image = load_image(image_file_tmp)
	results = model.detect([image])		# inference, results is a dictionary
	if len(results) == 0: 
		count += 1
		print_log('Mask-RCNN demo: testing %d/%d, no detected results!!!!!' % (count, num_list), log=log_file)
		continue

	# visualize and save results
	r = results[0]			# results from the first image
	num_instances = r['masks'].shape[-1]
	bboxes_tmp = r['rois']			# y1, x1, y2, x2 format
	bboxes_tmp[:, [0, 1]] = bboxes_tmp[:, [1, 0]]
	bboxes_tmp[:, [2, 3]] = bboxes_tmp[:, [3, 2]]			# x1, y1, x2, y2 format	

	fig, _ = visualize_image_with_bbox_mask(image, boxes=bboxes_tmp, masks=r['masks'], class_ids=r['class_ids'], class_names=class_names_bg, scores=r['scores'], class_to_plot=[1])
	# save_path_tmp = os.path.join(vis_dir, filename+'.jpg')
	save_path_tmp = os.path.join(vis_dir, video_dir, filename+'.jpg')
	# visualize_image(image, save_path=save_path_tmp)
	save_vis_close_helper(fig=fig, transparent=False, save_path=save_path_tmp)

	elapsed = timer.toc(average=False)
	remaining_str = convert_secs2time((elapsed) / count * (num_list - count))
	elapsed_str = convert_secs2time(elapsed)
	print_str = 'Mask-RCNN demo: testing %d/%d, detected %d instances, EP: %s, ETA: %s' % (count, num_list, num_instances, elapsed_str, remaining_str)
	print(print_str)
	print_log('%s, saving to %s' % (print_str, filename), log=log_file, display=False)

	bbox_savedir = os.path.join(save_dir, 'bboxes'); mkdir_if_missing(bbox_savedir)
	# bbox_savepath = os.path.join(bbox_savedir, filename+'.txt'); bbox_savefile = open(bbox_savepath, 'w')
	bbox_savepath = os.path.join(bbox_savedir, video_dir, filename+'.txt'); mkdir_if_missing(bbox_savepath); bbox_savefile = open(bbox_savepath, 'w')

	# save data for each individual instances
	for instance_index in range(num_instances):
		class_tmp = r['class_ids'][instance_index]
		if class_tmp != 1: continue						# only detect the pedestrians

		bbox_tmp = bboxes_tmp[instance_index, :]		# TLBR format
		score_tmp = r['scores'][instance_index]

		mask_tmp = r['masks'][:, :, instance_index]
		mask_tmp *= 255
		# mask_dir_frame = os.path.join(mask_dir, filename); mkdir_if_missing(mask_dir_frame)
		mask_dir_frame = os.path.join(mask_dir, video_dir, filename); mkdir_if_missing(mask_dir_frame)
		save_path_tmp = os.path.join(mask_dir_frame, 'instance_%04d'%instance_index+'.jpg')		
		save_image(mask_tmp, save_path=save_path_tmp)
		
		# save info for every instances
		save_str = '%s %s %.2f %.2f %.2f %.2f %.2f %s \n' % (image_file_tmp, class_names_bg[class_tmp], bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3], score_tmp, save_path_tmp)
		detection_results_file.write(save_str)

		save_str = 'Pedestrian -1 -1 -10 %.3f %.3f %.3f %.3f 0 0 0 0 0 0 0 %.8f\n' % (bbox_tmp[0], bbox_tmp[1], bbox_tmp[2], bbox_tmp[3], score_tmp)
		bbox_savefile.write(save_str)
	
	# zxc
	count += 1
	bbox_savefile.close()

detection_results_file.close()