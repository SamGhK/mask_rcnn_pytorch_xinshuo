# # Author: Xinshuo Weng
# # email: xinshuo.weng@gmail.com

# import os, torch, numpy as np, matplotlib.pyplot as plt
# from config import Config
# if "DISPLAY" not in os.environ: plt.switch_backend('agg')
# from mylibs import MaskRCNN, class_names
# from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing, load_image, save_image
# from xinshuo_visualization import visualize_image_with_bbox_mask
# from xinshuo_visualization.python.private import save_vis_close_helper
# from xinshuo_miscellaneous import convert_secs2time, Timer

# train_dataset = 'coco'
# object_interest = ['Pedestrian', 'Car', 'Cyclist']
# # data_dir = '/media/xinshuo/Data/Datasets/KITTI/object/training'
# data_dir = '/media/xinshuo/Data/Datasets/Cityscapes/leftImg8bit'
# results_name = '2d_bbox_mask_detection_results_%s_' % (train_dataset)
# root_dir = os.getcwd()                      # Root directory of the project

# # Index of the class in the list is its ID. For example, to get ID of # the teddy bear class, use: class_names.index('teddy bear')
# if dataset == 'coco':
# 	class_names_bg = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
# 		'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
# 		'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
# 		'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
# 		'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
# 		'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
# 		'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
# 		'teddy bear', 'hair drier', 'toothbrush']
# elif dataset == 'cityscape': class_names_bg = ['BG'] + class_names
# else: assert False, 'error'

# class InferenceConfig(Config):
#     # Set batch size to 1 since we'll be running inference on
#     # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
#     NAME = 'evaluate_%s' % dataset
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     # DETECTION_MIN_CONFIDENCE = 0
#     if dataset == 'coco': NUM_CLASSES = 1 + 80
#     else: NUM_CLASSES = 1 + len(class_names)
# config = InferenceConfig()

# log_dir = os.path.join(root_dir, 'logs')    # Directory to save logs and trained model

# ##--------------------------------- Model Directory ----------------------------------##
# # model_path = os.path.join(root_dir, 'resnet50_imagenet.pth')    		# Path to trained weights from Imagenet
# model_path = os.path.join(root_dir, 'mask_rcnn_coco.pth')    			# Path to trained weights file
# # model_path = '/media/xinshuo/Data/models/mask_rcnn_pytorch/cityscape20181018T0035/mask_rcnn_cityscape_0060.pth'

# ##--------------------------------- Data Directory ----------------------------------##
# images_dir = os.path.join(data_dir, 'val/frankfurt')
# save_dir = os.path.join(data_dir, 'results/%s' % results_name); mkdir_if_missing(save_dir)

# # save_dir = os.path.join(data_dir, 'results/mask_preprocessed_cityscape'); mkdir_if_missing(save_dir)