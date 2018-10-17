# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# CityScape dataset loader

import os, time, numpy as np, copy
from cityscapesscripts.helpers.annotation import Annotation
from mylibs import General_Dataset, id2label, name2label, class_names
from config import Config
from xinshuo_io import mkdir_if_missing, fileparts, load_list_from_folder
from xinshuo_math import bboxes_from_mask
from xinshuo_miscellaneous import is_path_exists, islist, find_unique_common_from_lists
from xinshuo_visualization import visualize_image_with_bbox_mask
from xinshuo_visualization.python.private import save_vis_close_helper
try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

# ignore_id_list = [0, 1, 2, 3, 4, -1]

############################################################
#  Configurations
############################################################
class CityscapeConfig(Config):
    """Configuration for training on CityScape.
    """
    NAME = "cityscape"          # Give the configuration a recognizable name
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 12        # Number of classes (including background)

############################################################
#  Dataset Loader
############################################################
class CityScapeDataset(General_Dataset):
    def __init__(self, dataset_dir, split, gttype):
        '''
        create a subset of the CityScape dataset

        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val)
        '''
        super(CityScapeDataset, self).__init__()
        assert split == 'train' or split == 'val', 'the subset can be either train or val'
        assert gttype == 'gtFine' or gttype == 'gtCoarse', 'the type of gt data is not good'

        self.dataset_dir = dataset_dir
        self.split = split
        self.gttype = gttype

        self.image_dir = os.path.join(dataset_dir, 'leftImg8bit', split)
        self.anno_dir = os.path.join(dataset_dir, gttype, split)
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
        sweep the data and return the dictionary of infomation for every images

        outputs:
            images_dict:        a dictionary of info, keys are the image ids, values are a dictionary {'width':, 'height':, 'ids': a list of object ids contained}
        '''
        print('loading annotations into memory from %s...' % self.anno_dir)
        anno_list, num_anno = load_list_from_folder(self.anno_dir, ext_filter=['.json'], recursive=True, depth=2)
        print('number of annotations has been loaded: %d' % num_anno)
        images_dict = {}
        for anno_tmp in anno_list:
            _, filename, _ = fileparts(anno_tmp)
            image_id = filename.split('_gt')[0]
            anno_data = Annotation()
            anno_data.fromJsonFile(anno_tmp)
            obj_type_list = []
            for obj in anno_data.objects:
                label = obj.label

                if obj.deleted: continue            # If the object is deleted, skip it

                # if the label is not known, but ends with a 'group' (e.g. cargroup) try to remove the s and see if that works
                # also we know that this polygon describes a group
                # TODO: ?????? what is the group
                isGroup = False
                if (not label in name2label) and label.endswith('group'):
                    label = label[:-len('group')]
                    isGroup = True
                if not label in name2label:
                    # printError('Label {} not known.'.format(label))
                    continue
                labelTuple = name2label[label]          # the label tuple

                # get the class ID
                if encoding == 'ids': id_tmp = labelTuple.id
                elif encoding == 'trainIds': id_tmp = labelTuple.trainId

                # if id_tmp in ignore_id_list: continue
                obj_type_list.append(id_tmp)

            obj_type_list = list(set(obj_type_list))
            image_data = {'width': anno_data.imgWidth, 'height': anno_data.imgHeight, 'ids': obj_type_list}
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
            class_ids = sorted(id2label.keys())    
            image_ids = sorted(self.images_dict.keys())  # All images
            # print(image_ids[0: 5])
        print('number of images for the requested id added to the dataset is %d' % len(image_ids))

        # add all images and classes into the dataset
        for id_tmp in class_ids: self.add_class('cityscape', id_tmp, id2label[id_tmp].name)          # Add classes
        for id_tmp in image_ids:             # Add images
            city_tmp = id_tmp.split('_')[0]
            self.add_image('cityscape', image_id=id_tmp, path=os.path.join(self.image_dir, city_tmp, id_tmp+'_leftImg8bit.png'), width=self.images_dict[id_tmp]['width'], 
                height=self.images_dict[id_tmp]['height'], annotation_file=os.path.join(self.anno_dir, city_tmp, id_tmp+'_%s_polygons.json' % self.gttype))

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
        if image_info_tmp['source'] != 'cityscape': return super(CityScapeDataset, self).load_mask(image_index)

        annotation_file = image_info_tmp['annotation_file']
        # print(type(annotation_file))

        return self.anno2mask(annotation_file, image_index)

    def anno2mask(self, annotation_file, image_index, encoding='ids'):
        '''
        convert an annotation file from Cityscapes to an array of mask images and its corresponding ids

        parameters:
            annotation_file:        a path to the annotation file corresponding to an image

        outputs:
            masks:                  A uint8 array of shape [height, width, num_instances] with one mask per instance.
            class_ids:              a 1D int32 array of class IDs of the instance masks.
        '''
        anno_data = Annotation()
        anno_data.fromJsonFile(annotation_file)
        size = (anno_data.imgWidth, anno_data.imgHeight)          # the size of the image

        # the background
        # if encoding == 'ids': backgroundId = name2label['unlabeled'].id
        # elif encoding == 'trainIds': backgroundId = name2label['unlabeled'].trainId
        # else:
            # print("Unknown encoding '{}'".format(encoding))
            # return None

        masks = []
        class_ids = []
        for obj in anno_data.objects:
            instanceImg = Image.new('I', size, 0)        # this is the image that we want to create
            drawer = ImageDraw.Draw(instanceImg)                  # a drawer to draw into the image
            label   = obj.label
            polygon = obj.polygon
            if obj.deleted: continue            # If the object is deleted, skip it

            # if the label is not known, but ends with a 'group' (e.g. cargroup) try to remove the s and see if that works
            # also we know that this polygon describes a group
            # TODO: ?????? what is the group
            isGroup = False
            if (not label in name2label) and label.endswith('group'):
                label = label[:-len('group')]
                isGroup = True
            if not label in name2label: 
                # printError( "Label '{}' not known.".format(label) )
                continue
            labelTuple = name2label[label]          # the label tuple

            # get the class ID
            if encoding == "ids": id_tmp = labelTuple.id
            elif encoding == "trainIds": id_tmp = labelTuple.trainId
            
            if id_tmp < 0: continue         # If the ID is negative that polygon should not be drawn
            try:
                drawer.polygon(polygon, fill=1)
                mask_instance_tmp = np.array(instanceImg, dtype='uint8')
                masks.append(mask_instance_tmp)
                class_ids.append(id_tmp)
            except:
                print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
                raise

        if class_ids:
            masks = np.stack(masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        else: return super(CityScapeDataset, self).load_mask(image_index)           # TODO, what happened here

    def visualization(self, image_index, save_dir):
        '''
        visualize the images loaded into the dataset
        '''
        image_path = self.image_info[image_index]['path']
        _, filename, _ = fileparts(image_path)
        image = self.load_image(image_index)

        masks, class_ids = self.load_mask(image_index)
        bbox = bboxes_from_mask(masks)                           # (num_instances x 4)

        # print(type(name2label))
        # print(type(name2label.keys()))
        # print(type(name2label.keys().tolist()))
        # zxc
        # class_names = ['BG'] + class_names
        # print(class_names)
        # zxc
        fig, _ = visualize_image_with_bbox_mask(image, boxes=bbox, masks=masks, class_ids=class_ids, class_names=['BG'] + class_names)
        save_path_tmp = os.path.join(save_dir, filename+'.jpg')
        save_vis_close_helper(fig=fig, transparent=False, save_path=save_path_tmp)