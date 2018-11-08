#!/usr/bin/python
#
# Cityscapes labels
#
from collections import namedtuple
import os, json, datetime, locale
Point = namedtuple('Point', ['x', 'y'])         # A point in a polygon
from abc import ABCMeta, abstractmethod

# a label and all meta information
Label = namedtuple('Label', [
    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class
    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.
    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!
    'category'    , # The name of the category that this label belongs to
    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.
    'hasInstances', # Whether this label distinguishes between single instances or not
    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not
    'color'       , # The color of this label
    ])

#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#     Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#     Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#     Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#     Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#     Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#     Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#     Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#     Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
# ]

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'person'               , 1 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 2 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 3 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 4 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 5 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 6 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 7 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 8 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 9 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 10 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
]

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'person'               , 1 ,        11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 2 ,        12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 3 ,        13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 4 ,        14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 5 ,        15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 6 ,       255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 7 ,       255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 8 ,        16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 9 ,        17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 10 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'license plate'        , 11 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
#     Label(  'wall'                 , 12 ,         3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 13 ,         4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 14 ,       255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 15 ,       255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 16 ,       255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'pole'                 , 17 ,         5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#     Label(  'polegroup'            , 18 ,       255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#     Label(  'traffic light'        , 19 ,         6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 20 ,         7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
# ]

# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'person'               , 1 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 2 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 3 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
# ]

cityscape_class_names = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle']
# cityscape_class_names = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', \
    # 'licensy plate', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign']
# cityscape_class_names = ['person', 'rider', 'car']

cityscape_name2label = { label.name: label for label in labels}     # name to label object
cityscape_id2label   = { label.id  : label for label in labels}    # id to label object



# labels = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Label(  'person'               , 1 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#     Label(  'rider'                , 2 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#     Label(  'car'                  , 3 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#     Label(  'truck'                , 4 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#     Label(  'bus'                  , 5 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#     Label(  'caravan'              , 6 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#     Label(  'trailer'              , 7 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#     Label(  'train'                , 8 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#     Label(  'motorcycle'           , 9 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#     Label(  'bicycle'              , 10 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#     Label(  'traffic light'        , 11 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#     Label(  'traffic sign'         , 12 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#     Label(  'vegetation'           , 13 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#     Label(  'building'             , 14 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#     Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#     Label(  'fence'                , 16 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#     Label(  'guard rail'           , 17 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#     Label(  'bridge'               , 18 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#     Label(  'tunnel'               , 19 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#     Label(  'terrain'              , 20 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#     Label(  'sky'                  , 21 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
# ]

# cityscape_class_names = ['person', 'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle', 'traffic light', \ 
    # 'traffic sign', 'vegetation', 'building', 'wall', 'fence', 'guard rail', 'bridge', 'tunnel', 'terrain', 'sky']

def class_mapping_cityscape_to_kitti(class_id):
    if class_id in [1]: return 1        # pedestrian
    elif class_id in [2]: return 3        # cyclist
    elif class_id in [3]: return 2        # car
    else: return 0

# Type of an object
class CsObjectType():
    POLY = 1 # polygon
    BBOX = 2 # bounding box

# Abstract base class for annotation objects
class CsObject:
    __metaclass__ = ABCMeta

    def __init__(self, objType):
        self.objectType = objType
        # the label
        self.label    = ""

        # If deleted or not
        self.deleted  = 0
        # If verified or not
        self.verified = 0
        # The date string
        self.date     = ""
        # The username
        self.user     = ""
        # Draw the object
        # Not read from or written to JSON
        # Set to False if deleted object
        # Might be set to False by the application for other reasons
        self.draw     = True

    @abstractmethod
    def __str__(self): pass

    @abstractmethod
    def fromJsonText(self, jsonText, objId=-1): pass

    @abstractmethod
    def toJsonText(self): pass

    def updateDate( self ):
        try:
            locale.setlocale( locale.LC_ALL , 'en_US' )
        except locale.Error:
            locale.setlocale( locale.LC_ALL , 'us_us' )
        except:
            pass
        self.date = datetime.datetime.now().strftime("%d-%b-%Y %H:%M:%S")

    # Mark the object as deleted
    def delete(self):
        self.deleted = 1
        self.draw    = False

# Class that contains the information of a single annotated object as polygon
class CsPoly(CsObject):
    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.POLY)
        # the polygon as list of points
        self.polygon    = []
        # the object ID
        self.id         = -1

    def __str__(self):
        polyText = ""
        if self.polygon:
            if len(self.polygon) <= 4:
                for p in self.polygon:
                    polyText += '({},{}) '.format( p.x , p.y )
            else:
                polyText += '({},{}) ({},{}) ... ({},{}) ({},{})'.format(
                    self.polygon[ 0].x , self.polygon[ 0].y ,
                    self.polygon[ 1].x , self.polygon[ 1].y ,
                    self.polygon[-2].x , self.polygon[-2].y ,
                    self.polygon[-1].x , self.polygon[-1].y )
        else:
            polyText = "none"
        text = "Object: {} - {}".format( self.label , polyText )
        return text

    def fromJsonText(self, jsonText, objId):
        self.id = objId
        self.label = str(jsonText['label'])
        self.polygon = [ Point(p[0],p[1]) for p in jsonText['polygon'] ]
        if 'deleted' in jsonText.keys():
            self.deleted = jsonText['deleted']
        else:
            self.deleted = 0
        if 'verified' in jsonText.keys():
            self.verified = jsonText['verified']
        else:
            self.verified = 1
        if 'user' in jsonText.keys():
            self.user = jsonText['user']
        else:
            self.user = ''
        if 'date' in jsonText.keys():
            self.date = jsonText['date']
        else:
            self.date = ''
        if self.deleted == 1:
            self.draw = False
        else:
            self.draw = True

    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['id'] = self.id
        objDict['deleted'] = self.deleted
        objDict['verified'] = self.verified
        objDict['user'] = self.user
        objDict['date'] = self.date
        objDict['polygon'] = []
        for pt in self.polygon:
            objDict['polygon'].append([pt.x, pt.y])

        return objDict

# Class that contains the information of a single annotated object as bounding box
class CsBbox(CsObject):
    # Constructor
    def __init__(self):
        CsObject.__init__(self, CsObjectType.BBOX)
        # the polygon as list of points
        self.bbox  = []
        self.bboxVis  = []

        # the ID of the corresponding object
        self.instanceId = -1

    def __str__(self):
        bboxText = ""
        bboxText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format( 
            self.bbox[0] , self.bbox[1] ,  self.bbox[2] ,  self.bbox[3] )

        bboxVisText = ""
        bboxVisText += '[(x1: {}, y1: {}), (w: {}, h: {})]'.format( 
            self.bboxVis[0] , self.bboxVis[1] , self.bboxVis[2], self.bboxVis[3] )

        text = "Object: {} - bbox {} - visible {}".format( self.label , bboxText, bboxVisText )
        return text

    def fromJsonText(self, jsonText, objId=-1):
        self.bbox = jsonText['bbox']
        self.bboxVis = jsonText['bboxVis']
        self.label = str(jsonText['label'])
        self.instanceId = jsonText['instanceId']
    
    def toJsonText(self):
        objDict = {}
        objDict['label'] = self.label
        objDict['instanceId'] = self.instanceId
        objDict['bbox'] = self.bbox
        objDict['bboxVis'] = self.bboxVis

        return objDict

# The annotation of a whole image (doesn't support mixed annotations, i.e. combining CsPoly and CsBbox)
class CityScapeAnnotation:
    # Constructor
    def __init__(self, objType=CsObjectType.POLY):
        # the width of that image and thus of the label image
        self.imgWidth  = 0
        # the height of that image and thus of the label image
        self.imgHeight = 0
        # the list of objects
        self.objects = []
        assert objType in CsObjectType.__dict__.values()
        self.objectType = objType

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def fromJsonText(self, jsonText):
        jsonDict = json.loads(jsonText)
        self.imgWidth  = int(jsonDict['imgWidth'])
        self.imgHeight = int(jsonDict['imgHeight'])
        self.objects   = []
        for objId, objIn in enumerate(jsonDict[ 'objects' ]):
            if self.objectType == CsObjectType.POLY:
                obj = CsPoly()
            elif self.objectType == CsObjectType.BBOX:
                obj = CsBbox()
            obj.fromJsonText(objIn, objId)
            self.objects.append(obj)

    def toJsonText(self):
        jsonDict = {}
        jsonDict['imgWidth'] = self.imgWidth
        jsonDict['imgHeight'] = self.imgHeight
        jsonDict['objects'] = []
        for obj in self.objects:
            objDict = obj.toJsonText()
            jsonDict['objects'].append(objDict)
  
        return jsonDict

    # Read a json formatted polygon file and return the annotation
    def fromJsonFile(self, jsonFile):
        if not os.path.isfile(jsonFile):
            print('Given json file not found: {}'.format(jsonFile))
            return
        with open(jsonFile, 'r') as f:
            jsonText = f.read()
            self.fromJsonText(jsonText)

    def toJsonFile(self, jsonFile):
        with open(jsonFile, 'w') as f:
            f.write(self.toJson())
