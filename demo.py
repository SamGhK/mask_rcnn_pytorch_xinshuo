# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import os, coco, torch, numpy as np
from mylibs import MaskRCNN, display_instances
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing, load_image, save_image
from xinshuo_visualization.python.private import save_vis_close_helper

class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	# GPU_COUNT = 0 for CPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

root_dir = os.getcwd()                      # Root directory of the project
log_dir = os.path.join(root_dir, 'logs')    # Directory to save logs and trained model
model_path = os.path.join(root_dir, 'mask_rcnn_coco.pth')    # Path to trained weights file
# images_dir = os.path.join(root_dir, 'images')    # Directory of images to run detection on
data_dir = '/media/xinshuo/Data/Datasets/KITTI/training'
images_dir = os.path.join(data_dir, 'image_2')
save_dir = os.path.join(data_dir, 'mask_rcnn_seg_results'); mkdir_if_missing(save_dir)
vis_dir = os.path.join(save_dir, 'visualization'); mkdir_if_missing(vis_dir)
mask_dir = os.path.join(save_dir, 'masks'); mkdir_if_missing(mask_dir)

config = InferenceConfig()
config.display()

# Create model object.
model = MaskRCNN(model_dir=log_dir, config=config)
if config.GPU_COUNT: model = model.cuda()
model.load_state_dict(torch.load(model_path))    # Load weights trained on MS-COCO

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#                'bus', 'train', 'truck']

# load the data
image_list, num_list = load_list_from_folder(images_dir)
print('testing results on %d images' % num_list) 
for image_file_tmp in image_list:
	_, filename, ext = fileparts(image_file_tmp)
	
	image = load_image(image_file_tmp)
	results = model.detect([image])		# inference, results is a dictionary

	# visualize and save results
	r = results[0]			# results from the first image
	fig, _ = display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])

	save_path_tmp = os.path.join(vis_dir, filename+ext)
	print('saving to %s' % save_path_tmp)
	save_vis_close_helper(fig=fig, transparent=False, save_path=save_path_tmp)

	# save individual mask map
	num_instances = r['masks'].shape[-1]
	for instance_index in range(num_instances):
		mask_tmp = r['masks'][:, :, instance_index]
		
		mask_tmp *= 255
		mask_dir_frame = os.path.join(mask_dir, filename); mkdir_if_missing(mask_dir_frame)
		save_path_tmp = os.path.join(mask_dir_frame, 'instance_%04d'%instance_index+ext)
		
		save_image(mask_tmp, save_path=save_path_tmp)