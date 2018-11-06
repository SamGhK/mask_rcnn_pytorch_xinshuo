# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# CityScape dataset loader

import os, time, numpy as np, copy, cv2
from mylibs import General_Dataset, Config, kitti_class_names, kitti_id2label, class_mapping_kitti_to_mykitti
from xinshuo_io import mkdir_if_missing, fileparts, load_list_from_folder, load_image
from xinshuo_math import bboxes_from_mask
from xinshuo_images import rgb2gray
from xinshuo_miscellaneous import is_path_exists, islist, find_unique_common_from_lists
from xinshuo_visualization import visualize_image_with_bbox_mask, visualize_image
from xinshuo_visualization.python.private import save_vis_close_helper
try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

############################################################
#  Configurations
############################################################
class KITTIConfig(Config):
    """Configuration for training on CityScape.
    """
    NAME = 'kitti'          # Give the configuration a recognizable name
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + len(kitti_class_names)        # Number of classes (including background)

############################################################
#  Dataset Loader
############################################################
class KITTIDataset(General_Dataset):
    def __init__(self, dataset_dir, split):
        '''
        create a subset of the CityScape dataset

        dataset_dir: The root directory of the dataset.
        subset: What to load (training, testing)
        '''
        super(KITTIDataset, self).__init__()
        assert split == 'training' or split == 'testing', 'the subset can be either training or testing'
        self.dataset_dir = dataset_dir
        self.split = split

        self.image_dir = os.path.join(dataset_dir, split, 'image_2')
        self.anno_dir = os.path.join(dataset_dir, split, 'instance')
        self.images_dict = self.sweep_data()

    def get_image_ids(self, class_ids):
        '''
        return a list of image IDs containing the given class IDs

        parameters:
            class_ids:          a list of class IDs requested

        outputs:
            image_ids:          a list of image IDs containing the requested classes
        '''
        assert islist(class_ids), 'the input class ids is not a list'
        image_id_list = []
        for image_id_tmp, image_data in self.images_dict.items():
            obj_ids_tmp = image_data['ids']
            intersect_list = find_unique_common_from_lists(class_ids, obj_ids_tmp)
            if len(intersect_list) > 0: image_id_list.append(image_id_tmp)
        return image_id_list

    def sweep_data(self, encoding='ids'):
        '''
        sweep the data and return the dictionary of ground truth infomation for every images

        outputs:
            images_dict:        a dictionary of info, keys are the image ids, values are a dictionary {'width':, 'height':, 'ids': a list of object ids contained}
        '''
        print('loading annotations into memory from %s...' % self.anno_dir)
        anno_list, num_anno = load_list_from_folder(self.anno_dir, ext_filter=['.png'], recursive=True, depth=2)
        print('number of annotations has been loaded: %d' % num_anno)
        images_dict = {}
        for anno_tmp in anno_list:
            _, image_id, _ = fileparts(anno_tmp)
            img_shape, class_ids = self.anno2mask(anno_tmp, run_fast=True)
            class_ids = list(set(class_ids))
            image_data = {'width': img_shape[0], 'height': img_shape[1], 'ids': class_ids}
            images_dict[image_id] = image_data
            
        return images_dict

    def load_data(self, class_ids=None):
        '''
        load the data from the subset, basically to create the list of images and list of class ids

        parameters:
            class_ids:              If provided, only loads images that have the given classes.

        outputs:
            self.add_class:         add the requested class ids to the dataset
            self.add_image:         add the requested images to the dataset
        '''
        # print(self.images_dict.keys())
        # zxc
        
        if class_ids is not None:       # All images or a subset?
            image_ids = []
            for id_tmp in class_ids: image_ids.extend(self.get_image_ids(class_ids=[id_tmp]))
            image_ids = list(set(image_ids))        # Remove `1duplicates
        else: 
            print('loading all data')
            class_ids = sorted(kitti_id2label.keys())    
            image_ids = sorted(self.images_dict.keys())  # All images
        print('number of images for the requested id added to the dataset is %d' % len(image_ids))

        # add all images and classes into the dataset
        for id_tmp in class_ids: self.add_class('kitti', id_tmp, kitti_id2label[id_tmp].name)          # Add classes
        for id_tmp in image_ids:             # Add images
            self.add_image('kitti', image_id=id_tmp, path=os.path.join(self.image_dir, id_tmp+'.png'), width=self.images_dict[id_tmp]['width'], 
                height=self.images_dict[id_tmp]['height'], annotation_file=os.path.join(self.anno_dir, id_tmp+'.png'))

    def load_mask(self, image_index):
        '''
        load instance masks and class ids for the given image

        parameters:
            image_index:            an index to the image in the dataset, from 0 to num_data

        outputs:
            masks:                  A uint8 array of shape [height, width, num_instances] with one mask per instance.
            class_ids:              a 1D int32 array of class IDs of the instance masks.
        '''
        
        image_info_tmp = self.image_info[image_index]
        if image_info_tmp['source'] != 'kitti': return super(KITTIDataset, self).load_mask(image_index)

        annotation_file = image_info_tmp['annotation_file']
        return self.anno2mask(annotation_file, image_index)

    def anno2mask(self, annotation_file, image_index=0, run_fast=False):
        '''
        convert an annotation file from KITTI to an array of mask images and its corresponding ids

        parameters:
            annotation_file:        a path to the annotation file corresponding to an image

        outputs:
            masks:                  A uint8 array of shape [height, width, num_instances] with one mask per instance.
            class_ids:              a 1D int32 array of class IDs of the instance masks.
        '''
        label_img = np.array(Image.open(annotation_file))
        size = (label_img.shape[1], label_img.shape[0])         # width x height

        # Loop through all instance ids in instance image
        class_ids = []
        masks = []
        for pixel_id in np.unique(label_img):
            id_tmp = int(pixel_id // 256)
            id_tmp = class_mapping_kitti_to_mykitti(id_tmp)
            if not (id_tmp in kitti_id2label.keys()): continue              # ignore the class we do not care
            class_ids.append(id_tmp)

            if run_fast: continue
            mask_tmp = (label_img == pixel_id).astype('uint8')
            masks.append(mask_tmp)

        if run_fast: return size, class_ids
        if class_ids:
            masks = np.stack(masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        else: return super(KITTIDataset, self).load_mask(image_index)           # TODO, what happened here

    def visualization(self, image_index, save_dir):
        '''
        visualize the images loaded into the dataset
        '''
        image_path = self.image_info[image_index]['path']
        _, filename, _ = fileparts(image_path)
        image = self.load_image(image_index)

        masks, class_ids = self.load_mask(image_index)
        bbox = bboxes_from_mask(masks)                           # TLBR format, N x 4
        fig, _ = visualize_image_with_bbox_mask(image, boxes=bbox, masks=masks, class_ids=class_ids, class_names=['BG'] + kitti_class_names)
        save_path_tmp = os.path.join(save_dir, filename+'.jpg')
        save_vis_close_helper(fig=fig, transparent=False, save_path=save_path_tmp)