# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

# CityScape dataset loader

import os, time, numpy as np, zipfile, urllib.request, shutil
from mylibs import General_Dataset
from config import Config
from xinshuo_io import mkdir_if_missing
from xinshuo_miscellaneous import is_path_exists

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
#  Dataset
############################################################
class CityScapeDataset(General_Dataset):
    def load_data(self, dataset_dir, subset, class_ids=None):
        """
        Load a subset of the CityScape dataset
        dataset_dir: The root directory of the dataset.
        subset: What to load (train, val)
        class_ids: If provided, only loads images that have the given classes.
        """

        coco = COCO('{}/annotations/instances_{}{}.json'.format(dataset_dir, subset, year))
        if subset == 'minival' or subset == 'valminusminival': subset = 'val'
        image_dir = os.path.join(dataset_dir, '{}{}'.format(subset, year))

        # Load all classes or a subset?
        if not class_ids: class_ids = sorted(coco.getCatIds())        # All classes
        # print(class_ids)

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids: image_ids.extend(list(coco.getImgIds(catIds=[id])))
            image_ids = list(set(image_ids))        # Remove duplicates
        else: image_ids = list(coco.imgs.keys())  # All images

        # add all images and classes into the dataset
        for i in class_ids: self.add_class('coco', i, coco.loadCats(i)[0]['name'])          # Add classes
        for i in image_ids:             # Add images
            self.add_image('coco', image_id=i, path=os.path.join(image_dir, coco.imgs[i]['file_name']), width=coco.imgs[i]['width'], 
                height=coco.imgs[i]['height'], annotations=coco.loadAnns(coco.getAnnIds(imgIds=[i], catIds=class_ids, iscrowd=None)))
        
        if return_coco: return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco": return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id("coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"], image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1: continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else: return super(CocoDataset, self).load_mask(image_id)     # Call super class to return an empty mask

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco": return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else: super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            rle = maskUtils.frPyObjects(segm, height, width)    # uncompressed RLE
        else: rle = ann['segmentation']   # rle
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m
