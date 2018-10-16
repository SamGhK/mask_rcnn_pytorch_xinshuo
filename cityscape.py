# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# CityScape dataset loader

import os, time, numpy as np
from cityscapesscripts.helpers.labels import id2label, labels, name2label
from cityscapesscripts.helpers.annotation import Annotation
from mylibs import General_Dataset
from config import Config
from xinshuo_io import mkdir_if_missing
from xinshuo_miscellaneous import is_path_exists
try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

############################################################
#  Configurations
############################################################
class CityscapeConfig(Config):
    """Configuration for training on CityScape.
    """
    NAME = "cityscape"          # Give the configuration a recognizable name
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 30        # Number of classes (including background)

############################################################
#  Dataset Loader
############################################################
class CityScapeDataset(General_Dataset):
    def __init__(self, dataset_dir, split):
        '''
        create a subset of the CityScape dataset

        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val)
        '''
        super(CityScapeDataset, self).__init__()
        assert subset == 'train' or subset == 'val', 'the subset can be either train or val'

        self.dataset_dir = dataset_dir
        self.split = split
        self.image_dir = os.path.join(dataset_dir, 'leftImg8bit', subset)
        self.anno_dir = os.path.join(dataset_dir, 'gtFine', subset)
        del id2label[0]; del id2label[1]; del id2label[2]; del id2label[3]; del id2label[4]         # remove the unlabeled class
        self.id2label = id2label
        self.images_dict = self.sweep_data()

    def get_image_ids(self, class_ids):
        '''
        return a list of image IDs containing the given class IDs

        parameters:
            class_ids:          a list of class IDs requested

        outputs:
            image_ids:          a list of image IDs containing the requested classes
        '''

    def sweep_data(self):
        '''
        sweep the data and return the dictionary of infomation for every images

        outputs:
            images_dict:        a dictionary of info, keys are the image ids, values are a dictionary {''}
        '''
        print('loading annotations into memory...')
        anno_list, num_anno = load_list_from_folder(self.anno_dir, ext_filter=['.json'], depth=2)
        print('number of annotations has been loaded: %d' % num_anno)
        for anno_tmp in anno_list:
            anno_data = Annotation()
            anno_data.fromJsonFile(annotation_file)

    def load_data(self, class_ids=None):
        """
        load the data from thr subset, basically to create the list of images and list of class ids

        class_ids: If provided, only loads images that have the given classes.
        """

        # Load all classes or a subset?
        if not class_ids: class_ids = sorted(self.id2label.keys())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id_tmp in class_ids: image_ids.extend(self.get_image_ids(class_ids=[id_tmp]))
            image_ids = list(set(image_ids))        # Remove duplicates
        else: image_ids = list(self.images_dict.keys())  # All images

        # add all images and classes into the dataset
        for id_tmp in class_ids: self.add_class('cityscape', id_tmp, self.id2label[id_tmp].name)          # Add classes
        for id_tmp in image_ids:             # Add images
            city_tmp = id_tmp.split('_')[0]
            self.add_image('cityscape', image_id=i, path=os.path.join(self.image_dir, city_tmp, id_tmp+'_leftImg8bit.png'), width=self.images_dict[id_tmp]['width'], 
                height=self.images_dict[id_tmp]['height'], annotations_file=os.path.join(self.anno_dir, city_tmp, id_tmp+'_gtFine_polygons.json'))

    def load_mask(self, image_index):
        '''
        load instance masks and class ids for the given image

        parameters:
            image_index:            an index to the image in the dataset, from 0 to num_data

        outputs:
            masks:                  A bool array of shape [height, width, num_instances] with one mask per instance.
            class_ids:              a 1D int32 array of class IDs of the instance masks.
        '''
        
        image_info = self.image_info[image_id]
        if image_info['source'] != 'cityscape': return super(CityScapeDataset, self).load_mask(image_index)
        annotations_file = self.image_info[image_index]["annotations"]
        return self.anno2mask(annotation_file)

    def anno2mask(annotation_file, encoding='ids'):
        '''
        convert an annotation file from Cityscapes to an array of mask images and its corresponding ids

        parameters:
            annotation_file:        a path to the annotation file corresponding to an image

        outputs:
            masks:                  A bool array of shape [height, width, instance count] with one mask per instance.
            class_ids:              a 1D int32 array of class IDs of the instance masks.
        '''
        anno_data = Annotation()
        anno_data.fromJsonFile(annotation_file)
        size = (anno_data.imgWidth, anno_data.imgHeight)          # the size of the image

        # the background
        if encoding == 'ids': backgroundId = name2label['unlabeled'].id
        elif encoding == 'trainIds': backgroundId = name2label['unlabeled'].trainId
        else:
            print("Unknown encoding '{}'".format(encoding))
            return None

        masks = []
        class_ids = []
        for obj in anno_data.objects:
            instanceImg = Image.new('I', size, backgroundId)        # this is the image that we want to create
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
            if not label in name2label: printError( "Label '{}' not known.".format(label) )
            labelTuple = name2label[label]          # the label tuple

            # get the class ID
            if encoding == "ids": id_tmp = labelTuple.id
            elif encoding == "trainIds": id_tmp = labelTuple.trainId
            
            if id_tmp < 0: continue         # If the ID is negative that polygon should not be drawn
            try:
                drawer.polygon(polygon, fill=1)
                mask_instance_tmp = np.array(instanceImg, dtype='bool').reshape((size[1], size[0], 1))
                print(mask_instance_tmp.shape)
                masks.append(mask_instance_tmp)
                class_ids.append(id_tmp)
            except:
                print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
                raise

        if class_ids:
            masks = np.stack(masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return masks, class_ids
        else: return super(CityScapeDataset, self).load_mask(image_id)