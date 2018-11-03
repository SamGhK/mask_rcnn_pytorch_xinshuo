# Author: Xinshuo
# Email: xinshuow@andrew.cmu.edu

import torch, torch.nn as nn, torch.nn.functional as F, datetime, os, re, torch.optim as optim, numpy as np, warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

from torch.autograd import Variable
from .nms.nms_wrapper import nms
from .roialign.roi_align.crop_and_resize import CropAndResizeFunction
from .mask_rcnn_dataset import Mask_RCNN_Dataset
from .rpn import RPN
from .resnet import ResNet
from .fpn import FPN, Classifier, Mask
from .loss import compute_losses
from .visualize import plot_loss
from .general_utils import generate_pyramid_anchors, resize_image, mold_image, compose_image_meta, parse_image_meta
from .pytorch_myutils import unique1d, intersect1d

from xinshuo_math import unmold_mask
from xinshuo_miscellaneous import log, printProgressBar, is_path_exists, islist, print_log
from xinshuo_io import mkdir_if_missing

############################################################
#  Proposal Layer
############################################################
def apply_box_deltas(boxes, deltas):
	"""Applies the given deltas to the given boxes.
	boxes: [N, 4] where each row is y1, x1, y2, x2
	deltas: [N, 4] where each row is [dy, dx, log(dh), log(dw)]
	"""
	# Convert to y, x, h, w
	height = boxes[:, 2] - boxes[:, 0]
	width = boxes[:, 3] - boxes[:, 1]
	center_y = boxes[:, 0] + 0.5 * height
	center_x = boxes[:, 1] + 0.5 * width
	# Apply deltas
	center_y += deltas[:, 0] * height
	center_x += deltas[:, 1] * width
	height *= torch.exp(deltas[:, 2])
	width *= torch.exp(deltas[:, 3])
	# Convert back to y1, x1, y2, x2
	y1 = center_y - 0.5 * height
	x1 = center_x - 0.5 * width
	y2 = y1 + height
	x2 = x1 + width
	result = torch.stack([y1, x1, y2, x2], dim=1)
	return result

def clip_boxes(boxes, window):
	"""
	boxes: [N, 4] each col is y1, x1, y2, x2
	window: [4] in the form y1, x1, y2, x2
	"""
	boxes = torch.stack([boxes[:, 0].clamp(float(window[0]), float(window[2])), boxes[:, 1].clamp(float(window[1]), float(window[3])), 
		boxes[:, 2].clamp(float(window[0]), float(window[2])), boxes[:, 3].clamp(float(window[1]), float(window[3]))], 1)
	return boxes

def proposal_layer(inputs, proposal_count, nms_threshold, anchors, config=None):
	"""Receives anchor scores and selects a subset to pass as proposals
	to the second stage. Filtering is done based on anchor scores and
	non-max suppression to remove overlaps. It also applies bounding
	box refinment detals to anchors.

	Inputs:
	    rpn_probs: [batch, anchors, (bg prob, fg prob)]
	    rpn_bbox_refine: [batch, anchors, (dy, dx, log(dh), log(dw))]

	Returns:
	    Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
	"""
	inputs[0], inputs[1] = inputs[0].squeeze(0), inputs[1].squeeze(0)      # Currently only supports batchsize 1
	scores = inputs[0][:, 1]           # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]

	# Box deltas [batch, num_rois, 4]
	deltas = inputs[1]
	std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
	if config.GPU_COUNT: std_dev = std_dev.cuda()
	deltas = deltas * std_dev

	# Improve performance by trimming to top anchors by score
	# and doing the rest on the smaller subset.
	pre_nms_limit = min(6000, anchors.size()[0])
	scores, order = scores.sort(descending=True)
	order = order[:pre_nms_limit]
	scores = scores[:pre_nms_limit]
	deltas = deltas[order.data, :] # TODO: Support batch size > 1 ff.
	anchors = anchors[order.data, :]

	# Apply deltas to anchors to get refined anchors.
	# [batch, N, (y1, x1, y2, x2)]
	boxes = apply_box_deltas(anchors, deltas)

	# Clip to image boundaries. [batch, N, (y1, x1, y2, x2)]
	height, width = config.IMAGE_SHAPE[:2]
	window = np.array([0, 0, height, width]).astype(np.float32)
	boxes = clip_boxes(boxes, window)

	# Filter out small boxes
	# According to Xinlei Chen's paper, this reduces detection accuracy
	# for small objects, so we're skipping it.

	# Non-max suppression
	keep = nms(torch.cat((boxes, scores.unsqueeze(1)), 1).data, nms_threshold)
	keep = keep[:proposal_count]
	boxes = boxes[keep, :]

	# Normalize dimensions to range of 0 to 1.
	norm = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
	if config.GPU_COUNT: norm = norm.cuda()
	normalized_boxes = boxes / norm

	# Add back batch dimension
	normalized_boxes = normalized_boxes.unsqueeze(0)

	return normalized_boxes

############################################################
#  Detection Target Layer
############################################################
def bbox_overlaps(boxes1, boxes2):
	"""Computes IoU overlaps between two sets of boxes.
	boxes1, boxes2: [N, (y1, x1, y2, x2)].
	"""
	# 1. Tile boxes2 and repeate boxes1. This allows us to compare
	# every boxes1 against every boxes2 without loops.
	# TF doesn't have an equivalent to np.repeate() so simulate it
	# using tf.tile() and tf.reshape.
	boxes1_repeat = boxes2.size()[0]
	boxes2_repeat = boxes1.size()[0]
	boxes1 = boxes1.repeat(1,boxes1_repeat).view(-1,4)
	boxes2 = boxes2.repeat(boxes2_repeat,1)

	# 2. Compute intersections
	b1_y1, b1_x1, b1_y2, b1_x2 = boxes1.chunk(4, dim=1)
	b2_y1, b2_x1, b2_y2, b2_x2 = boxes2.chunk(4, dim=1)
	y1 = torch.max(b1_y1, b2_y1)[:, 0]
	x1 = torch.max(b1_x1, b2_x1)[:, 0]
	y2 = torch.min(b1_y2, b2_y2)[:, 0]
	x2 = torch.min(b1_x2, b2_x2)[:, 0]
	zeros = Variable(torch.zeros(y1.size()[0]), requires_grad=False)
	if y1.is_cuda: zeros = zeros.cuda()
	intersection = torch.max(x2 - x1, zeros) * torch.max(y2 - y1, zeros)

	# 3. Compute unions
	b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
	b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
	union = b1_area[:,0] + b2_area[:,0] - intersection

	# 4. Compute IoU and reshape to [boxes1, boxes2]
	iou = intersection / union
	overlaps = iou.view(boxes2_repeat, boxes1_repeat)

	return overlaps

def box_refinement(box, gt_box):
	"""Compute refinement needed to transform box to gt_box.
	box and gt_box are [N, (y1, x1, y2, x2)]
	"""

	height = box[:, 2] - box[:, 0]
	width = box[:, 3] - box[:, 1]
	center_y = box[:, 0] + 0.5 * height
	center_x = box[:, 1] + 0.5 * width

	gt_height = gt_box[:, 2] - gt_box[:, 0]
	gt_width = gt_box[:, 3] - gt_box[:, 1]
	gt_center_y = gt_box[:, 0] + 0.5 * gt_height
	gt_center_x = gt_box[:, 1] + 0.5 * gt_width

	dy = (gt_center_y - center_y) / height
	dx = (gt_center_x - center_x) / width
	dh = torch.log(gt_height / height)
	dw = torch.log(gt_width / width)

	result = torch.stack([dy, dx, dh, dw], dim=1)
	return result

def detection_target_layer(proposals, gt_class_ids, gt_boxes, gt_masks, config):
    """Subsamples proposals and generates target box refinment, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinments.
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width)
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.
    """

    # Currently only supports batchsize 1
    proposals, gt_class_ids, gt_boxes, gt_masks = proposals.squeeze(0), gt_class_ids.squeeze(0), gt_boxes.squeeze(0), gt_masks.squeeze(0)

    # Handle COCO crowds 
    # A crowd box in COCO is a bounding box around several instances. Exclude them from training. A crowd box is given a negative class ID.
    if torch.nonzero(gt_class_ids < 0).size()[0]:       
        crowd_ix = torch.nonzero(gt_class_ids < 0)[:, 0]        
        non_crowd_ix = torch.nonzero(gt_class_ids > 0)[:, 0]
        crowd_boxes = gt_boxes[crowd_ix.data, :]
        crowd_masks = gt_masks[crowd_ix.data, :, :]

        gt_class_ids = gt_class_ids[non_crowd_ix.data]
        gt_boxes = gt_boxes[non_crowd_ix.data, :]           # N x 4
        gt_masks = gt_masks[non_crowd_ix.data, :]           # N x h x w

        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = bbox_overlaps(proposals, crowd_boxes)
        crowd_iou_max = torch.max(crowd_overlaps, dim=1)[0]
        no_crowd_bool = crowd_iou_max < 0.001
    else:
        no_crowd_bool =  Variable(torch.ByteTensor(proposals.size()[0]*[True]), requires_grad=False)
        if config.GPU_COUNT: no_crowd_bool = no_crowd_bool.cuda()

    overlaps = bbox_overlaps(proposals, gt_boxes)       # Compute overlaps matrix [proposals, gt_boxes]
    roi_iou_max = torch.max(overlaps, dim=1)[0]         # Determine postive and negative ROIs
    positive_roi_bool = roi_iou_max >= 0.5              # 1. Positive ROIs are those with >= 0.5 IoU with a GT box

    # Subsample ROIs. Aim for 33% positive Positive ROIs
    if torch.nonzero(positive_roi_bool).size()[0]:
        positive_indices = torch.nonzero(positive_roi_bool)[:, 0]

        positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
        rand_idx = torch.randperm(positive_indices.size()[0])
        rand_idx = rand_idx[:positive_count]
        if config.GPU_COUNT: rand_idx = rand_idx.cuda()
        positive_indices = positive_indices[rand_idx]
        positive_count = positive_indices.size()[0]
        positive_rois = proposals[positive_indices.data,:]

        # Assign positive ROIs to GT boxes.
        positive_overlaps = overlaps[positive_indices.data,:]
        roi_gt_box_assignment = torch.max(positive_overlaps, dim=1)[1]
        roi_gt_boxes = gt_boxes[roi_gt_box_assignment.data,:]
        roi_gt_class_ids = gt_class_ids[roi_gt_box_assignment.data]

        # Compute bbox refinement for positive ROIs
        deltas = Variable(box_refinement(positive_rois.data, roi_gt_boxes.data), requires_grad=False)
        std_dev = Variable(torch.from_numpy(config.BBOX_STD_DEV).float(), requires_grad=False)
        if config.GPU_COUNT: std_dev = std_dev.cuda()
        deltas /= std_dev

        roi_masks = gt_masks[roi_gt_box_assignment.data,:,:]        # Assign positive ROIs to GT masks

        # Compute mask targets
        boxes = positive_rois
        if config.USE_MINI_MASK:
            # Transform ROI corrdinates from normalized image space to normalized mini-mask space.
            y1, x1, y2, x2 = positive_rois.chunk(4, dim=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = roi_gt_boxes.chunk(4, dim=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = torch.cat([y1, x1, y2, x2], dim=1)
        box_ids = Variable(torch.arange(roi_masks.size()[0]), requires_grad=False).int()
        if config.GPU_COUNT: box_ids = box_ids.cuda()
        masks = Variable(CropAndResizeFunction(config.MASK_SHAPE[0], config.MASK_SHAPE[1], 0)(roi_masks.unsqueeze(1), boxes, box_ids).data, requires_grad=False)
        masks = masks.squeeze(1)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with binary cross entropy loss.
        masks = torch.round(masks)
    else: positive_count = 0

    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_roi_bool = roi_iou_max < 0.5
    negative_roi_bool = negative_roi_bool & no_crowd_bool

    # Negative ROIs. Add enough to maintain positive:negative ratio.
    if torch.nonzero(negative_roi_bool).size()[0] and positive_count>0:
        negative_indices = torch.nonzero(negative_roi_bool)[:, 0]
        r = 1.0 / config.ROI_POSITIVE_RATIO
        negative_count = int(r * positive_count - positive_count)
        rand_idx = torch.randperm(negative_indices.size()[0])
        rand_idx = rand_idx[:negative_count]
        if config.GPU_COUNT: rand_idx = rand_idx.cuda()
        negative_indices = negative_indices[rand_idx]
        negative_count = negative_indices.size()[0]
        negative_rois = proposals[negative_indices.data, :]
    else: negative_count = 0

    # Append negative ROIs and pad bbox deltas and masks that are not used for negative ROIs with zeros.
    if positive_count > 0 and negative_count > 0:
        rois = torch.cat((positive_rois, negative_rois), dim=0)
        zeros = Variable(torch.zeros(negative_count), requires_grad=False).int()
        if config.GPU_COUNT: zeros = zeros.cuda()
        roi_gt_class_ids = torch.cat([roi_gt_class_ids, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count,4), requires_grad=False)
        if config.GPU_COUNT: zeros = zeros.cuda()
        deltas = torch.cat([deltas, zeros], dim=0)
        zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT: zeros = zeros.cuda()
        masks = torch.cat([masks, zeros], dim=0)
    elif positive_count > 0: rois = positive_rois
    elif negative_count > 0:
        rois = negative_rois
        zeros = Variable(torch.zeros(negative_count), requires_grad=False)
        if config.GPU_COUNT: zeros = zeros.cuda()
        roi_gt_class_ids = zeros
        zeros = Variable(torch.zeros(negative_count,4), requires_grad=False).int()
        if config.GPU_COUNT: zeros = zeros.cuda()
        deltas = zeros
        zeros = Variable(torch.zeros(negative_count,config.MASK_SHAPE[0],config.MASK_SHAPE[1]), requires_grad=False)
        if config.GPU_COUNT: zeros = zeros.cuda()
        masks = zeros
    else:
        rois = Variable(torch.FloatTensor(), requires_grad=False)
        roi_gt_class_ids = Variable(torch.IntTensor(), requires_grad=False)
        deltas = Variable(torch.FloatTensor(), requires_grad=False)
        masks = Variable(torch.FloatTensor(), requires_grad=False)
        if config.GPU_COUNT:
            rois = rois.cuda()
            roi_gt_class_ids = roi_gt_class_ids.cuda()
            deltas = deltas.cuda()
            masks = masks.cuda()

    return rois, roi_gt_class_ids, deltas, masks

############################################################
#  Detection Layer
############################################################

def clip_to_window(window, boxes):
    """
        window: (y1, x1, y2, x2). The window in the image we want to clip to.
        boxes: [N, (y1, x1, y2, x2)]
    """
    boxes[:, 0] = boxes[:, 0].clamp(float(window[0]), float(window[2]))
    boxes[:, 1] = boxes[:, 1].clamp(float(window[1]), float(window[3]))
    boxes[:, 2] = boxes[:, 2].clamp(float(window[0]), float(window[2]))
    boxes[:, 3] = boxes[:, 3].clamp(float(window[1]), float(window[3]))
    return boxes

def refine_detections(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)]
    """

    # Class IDs per ROI
    _, class_ids = torch.max(probs, dim=1)

    # Class probability of the top class of each ROI. Class-specific bounding box deltas
    idx = torch.arange(class_ids.size()[0]).long()
    if config.GPU_COUNT: idx = idx.cuda()
    class_scores = probs[idx, class_ids.data]
    deltas_specific = deltas[idx, class_ids.data]

    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    std_dev = Variable(torch.from_numpy(np.reshape(config.RPN_BBOX_STD_DEV, [1, 4])).float(), requires_grad=False)
    if config.GPU_COUNT: std_dev = std_dev.cuda()
    refined_rois = apply_box_deltas(rois, deltas_specific * std_dev)

    # Convert coordiates to image domain
    height, width = config.IMAGE_SHAPE[:2]
    scale = Variable(torch.from_numpy(np.array([height, width, height, width])).float(), requires_grad=False)
    if config.GPU_COUNT: scale = scale.cuda()
    refined_rois *= scale

    refined_rois = clip_to_window(window, refined_rois)			# Clip boxes to image window
    refined_rois = torch.round(refined_rois)				    # Round and cast to int since we're deadling with pixels now

    # TODO: Filter out boxes with zero area
    keep_bool = class_ids>0				# Filter out background boxes
    # print(keep_bool)

    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE: keep_bool = keep_bool & (class_scores >= config.DETECTION_MIN_CONFIDENCE)
    # print(keep_bool)
    # print(torch.nonzero(keep_bool))

    if torch.nonzero(keep_bool).size()[0]:
        keep = torch.nonzero(keep_bool)[:, 0] 

        # Apply per-class NMS
        pre_nms_class_ids = class_ids[keep.data]
        pre_nms_scores = class_scores[keep.data]
        pre_nms_rois = refined_rois[keep.data]
        for i, class_id in enumerate(unique1d(pre_nms_class_ids)):
            ixs = torch.nonzero(pre_nms_class_ids == class_id)[:,0]         # Pick detections of this class

            # Sort
            ix_rois = pre_nms_rois[ixs.data]
            ix_scores = pre_nms_scores[ixs]
            ix_scores, order = ix_scores.sort(descending=True)
            ix_rois = ix_rois[order.data,:]

            class_keep = nms(torch.cat((ix_rois, ix_scores.unsqueeze(1)), dim=1).data, config.DETECTION_NMS_THRESHOLD)
            class_keep = keep[ixs[order[class_keep].data].data]     # Map indicies
            if i==0: nms_keep = class_keep
            else: nms_keep = unique1d(torch.cat((nms_keep, class_keep)))

        keep = intersect1d(keep, nms_keep)

        # Keep top detections
        roi_count = config.DETECTION_MAX_INSTANCES
        top_ids = class_scores[keep.data].sort(descending=True)[1][:roi_count]
        keep = keep[top_ids.data]

        result = torch.cat((refined_rois[keep.data], class_ids[keep.data].unsqueeze(1).float(), class_scores[keep.data].unsqueeze(1)), dim=1)
    else:
        keep = torch.nonzero(keep_bool)
        result = keep.clone()
    # print(keep)

    # print(class_ids[keep.data])
    # class_ids[keep.data].unsqueeze(1).float()

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are in image domain.
    # result = torch.cat((refined_rois[keep.data], class_ids[keep.data].unsqueeze(1).float(), class_scores[keep.data].unsqueeze(1)), dim=1)
    # print(result)
    return result

def detection_layer(config, rois, mrcnn_class, mrcnn_bbox, image_meta):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_score)] in pixels
    """

    # Currently only supports batchsize 1
    rois = rois.squeeze(0)
    _, _, window, _ = parse_image_meta(image_meta)
    window = window[0]
    detections = refine_detections(rois, mrcnn_class, mrcnn_bbox, window, config)
    return detections

############################################################
#  MaskRCNN Class
############################################################
class MaskRCNN(nn.Module):
    def __init__(self, config, model_dir):
        """
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        super(MaskRCNN, self).__init__()
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.build(config=config)
        self.initialize_weights()
        self.loss_history = []
        self.val_loss_history = []

    def initialize_weights(self):
        """Initialize model weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform(m.weight)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_trainable(self, layer_regex, model=None, indent=0, verbose=1):
        """
        Sets model layers as trainable if their names match the given regular expression.
        """
        for param in self.named_parameters():
            layer_name = param[0]
            trainable = bool(re.fullmatch(layer_regex, layer_name))
            if not trainable: param[1].requires_grad = False

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        self.epoch = 0      # Set date and epoch counter as if starting a new model
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/mask\_rcnn\_\w+(\d{4})\.pth"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6))
        
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now)); mkdir_if_missing(self.log_dir)
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.pth".format(self.config.NAME.lower()))      # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{:04d}")
        self.log_filepath = os.path.join(self.log_dir, 'log.txt')
        self.log_file = open(self.log_filepath, 'w')

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names: return None, None        
        dir_name = os.path.join(self.model_dir, dir_names[-1])      # Pick last directory

        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints: return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        if is_path_exists(filepath):
            state_dict = torch.load(filepath)
            self.load_state_dict(state_dict, strict=False)
        else: print("Weight file not found ...")
        # self.set_log_dir(filepath)

    def build(self, config):
        """Build Mask R-CNN architecture.
        """

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        resnet = ResNet("resnet101", stage5=True)
        C1, C2, C3, C4, C5 = resnet.stages()

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        self.fpn = FPN(C1, C2, C3, C4, C5, out_channels=256)
        self.classifier = Classifier(256, config.POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)     # FPN Classifier
        self.mask = Mask(256, config.MASK_POOL_SIZE, config.IMAGE_SHAPE, config.NUM_CLASSES)        # FPN Mask

        # Generate Anchors
        self.anchors = Variable(torch.from_numpy(generate_pyramid_anchors(config.RPN_ANCHOR_SCALES, config.RPN_ANCHOR_RATIOS, 
            config.BACKBONE_SHAPES, config.BACKBONE_STRIDES, config.RPN_ANCHOR_STRIDE)).float(), requires_grad=False)
        if self.config.GPU_COUNT: self.anchors = self.anchors.cuda()
        self.rpn = RPN(len(config.RPN_ANCHOR_RATIOS), config.RPN_ANCHOR_STRIDE, 256)        # RPN

        # Fix batch norm layers
        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1: 
                for p in m.parameters(): p.requires_grad = False
        self.apply(set_bn_fix)

    def detect(self, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        molded_images, image_metas, windows = self.mold_inputs(images)      # Mold inputs to format expected by the neural network
        molded_images = torch.from_numpy(molded_images.transpose(0, 3, 1, 2)).float()   # Convert images to torch tensor
        if self.config.GPU_COUNT: molded_images = molded_images.cuda()
        molded_images = Variable(molded_images, volatile=True)      # Wrap in variable
        detections, mrcnn_mask = self.predict([molded_images, image_metas], mode='inference')       # Run object detection
        
        if detections.size()[0]:
            detections = detections.data.cpu().numpy()      # Convert to numpy
            mrcnn_mask = mrcnn_mask.permute(0, 1, 3, 4, 2).data.cpu().numpy()

            # Process detections
            results = []
            for i, image in enumerate(images):
                final_rois, final_class_ids, final_scores, final_masks = self.unmold_detections(detections[i], mrcnn_mask[i], image.shape, windows[i])
                results.append({"rois": final_rois, "class_ids": final_class_ids, "scores": final_scores, "masks": final_masks})
            return results
        else: return []

    def predict(self, input, mode):
        molded_images, image_metas = input[0], input[1]

        if mode == 'inference': self.eval()
        elif mode == 'training': 
            self.train()
            def set_bn_eval(m):     # Set batchnorm always in eval mode during training
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()
            self.apply(set_bn_eval)

        [p2_out, p3_out, p4_out, p5_out, p6_out] = self.fpn(molded_images)      # Feature extraction

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [p2_out, p3_out, p4_out, p5_out, p6_out]
        mrcnn_feature_maps = [p2_out, p3_out, p4_out, p5_out]

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps: layer_outputs.append(self.rpn(p))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        outputs = list(zip(*layer_outputs))
        outputs = [torch.cat(list(o), dim=1) for o in outputs]
        rpn_class_logits, rpn_class, rpn_bbox_refine = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.      
        proposal_count = self.config.POST_NMS_ROIS_TRAINING if mode == "training" else self.config.POST_NMS_ROIS_INFERENCE
        rpn_rois = proposal_layer([rpn_class, rpn_bbox_refine], proposal_count=proposal_count, nms_threshold=self.config.RPN_NMS_THRESHOLD, anchors=self.anchors, config=self.config)

        if mode == 'inference':
            # Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rpn_rois)     # Proposal classifier and BBox regressor heads
            detections = detection_layer(self.config, rpn_rois, mrcnn_class, mrcnn_bbox, image_metas)       # Detections, output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            if detections.size()[0]:
                # Convert boxes to normalized coordinates
                # TODO: let DetectionLayer return normalized coordinates to avoid
                #       unnecessary conversions
                h, w = self.config.IMAGE_SHAPE[:2]
                scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
                if self.config.GPU_COUNT: scale = scale.cuda()
                detection_boxes = detections[:, :4] / scale
                detection_boxes = detection_boxes.unsqueeze(0)  # Add back batch dimension
                mrcnn_mask = self.mask(mrcnn_feature_maps, detection_boxes)     # Create masks for detections

                # Add back batch dimension
                detections = detections.unsqueeze(0)
                mrcnn_mask = mrcnn_mask.unsqueeze(0)
            else: mrcnn_mask = detections.clone()
            return [detections, mrcnn_mask]
        elif mode == 'training':
            gt_class_ids, gt_boxes, gt_masks = input[2], input[3], input[4]

            # Normalize coordinates to [0, 1]
            h, w = self.config.IMAGE_SHAPE[:2]
            scale = Variable(torch.from_numpy(np.array([h, w, h, w])).float(), requires_grad=False)
            if self.config.GPU_COUNT: scale = scale.cuda()
            gt_boxes = gt_boxes / scale

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training. Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_deltas, target_mask = detection_target_layer(rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            if not rois.size()[0]:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask = Variable(torch.FloatTensor()), Variable(torch.IntTensor()), Variable(torch.FloatTensor()), Variable(torch.FloatTensor())
                if self.config.GPU_COUNT: mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask = mrcnn_class_logits.cuda(), mrcnn_class.cuda(), mrcnn_bbox.cuda(), mrcnn_mask.cuda()
            else:
                mrcnn_class_logits, mrcnn_class, mrcnn_bbox = self.classifier(mrcnn_feature_maps, rois)     # Proposal classifier and BBox regressor heads
                mrcnn_mask = self.mask(mrcnn_feature_maps, rois)                                            # Create masks for detections

            return [rpn_class_logits, rpn_bbox_refine, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask]

    def train_model(self, train_dataset, val_dataset, learning_rate, num_epochs, layers):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        num_epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # From a specific Resnet stage and up
            "3+": r"(fpn.C3.*)|(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "4+": r"(fpn.C4.*)|(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            "5+": r"(fpn.C5.*)|(fpn.P5\_.*)|(fpn.P4\_.*)|(fpn.P3\_.*)|(fpn.P2\_.*)|(rpn.*)|(classifier.*)|(mask.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys(): layers = layer_regex[layers]		# select the layers to train

        # Data generators
        train_set = Mask_RCNN_Dataset(train_dataset, self.config, augment=True)
        train_generator = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
        val_set = Mask_RCNN_Dataset(val_dataset, self.config, augment=True)
        val_generator = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

        # logging
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch+1, learning_rate), log=self.log_file)
        log("Checkpoint Path: {}".format(self.checkpoint_path), log=self.log_file)
        self.set_trainable(layers)			# only a selected number of layers are trained

        # Optimizer object, Add L2 Regularization, Skip gamma and beta weights of batch normalization layers.
        trainables_wo_bn = [param for name, param in self.named_parameters() if param.requires_grad and not 'bn' in name]
        trainables_only_bn = [param for name, param in self.named_parameters() if param.requires_grad and 'bn' in name]
        optimizer = optim.SGD([{'params': trainables_wo_bn, 'weight_decay': self.config.WEIGHT_DECAY}, {'params': trainables_only_bn}], lr=learning_rate, momentum=self.config.LEARNING_MOMENTUM)
        
        # train
        for epoch in range(self.epoch + 1, num_epochs + 1):
            log('Epoch {}/{}.'.format(epoch, num_epochs), log=self.log_file)

            loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask = self.train_epoch(train_generator, optimizer, self.config.STEPS_PER_EPOCH)     # Training
            val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask = self.valid_epoch(val_generator, self.config.VALIDATION_STEPS)     # Validation

            # Statistics
            self.loss_history.append([loss, loss_rpn_class, loss_rpn_bbox, loss_mrcnn_class, loss_mrcnn_bbox, loss_mrcnn_mask])
            self.val_loss_history.append([val_loss, val_loss_rpn_class, val_loss_rpn_bbox, val_loss_mrcnn_class, val_loss_mrcnn_bbox, val_loss_mrcnn_mask])
            plot_loss(self.loss_history, self.val_loss_history, save=True, log_dir=self.log_dir)
            torch.save(self.state_dict(), self.checkpoint_path.format(epoch))		# Save model

        self.epoch = num_epochs

    def train_epoch(self, datagenerator, optimizer, num_steps):
        batch_count = 0
        loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum = 0, 0, 0, 0, 0, 0
        step = 0
        optimizer.zero_grad()
        # print('I am here')
        for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, image_index, filename in datagenerator:
            # print(batch_count)
            # print(image_index.data)
            # print(image_index < 965)
            # if image_index.data != 1631: continue
            # print(image_index)
            # print(gt_class_ids)
            # print(type(gt_class_ids))
            if islist(gt_class_ids): continue           # TODO, why this happen
            # if islist(image_metas): print(gt_class_ids)
            # print(filename)

            # Progress
            printProgressBar(step + 1, num_steps, log=self.log_file, prefix="\t{}/{}".format(step + 1, num_steps), suffix='index: {:5d}, filename: {:40s}'.format(image_index.item(), filename[0]), length=10)

            batch_count += 1
            image_metas = image_metas.numpy()       # image_metas as numpy array
            images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = Variable(images), Variable(rpn_match), Variable(rpn_bbox), Variable(gt_class_ids), Variable(gt_boxes), Variable(gt_masks)
            if self.config.GPU_COUNT: images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = images.cuda(), rpn_match.cuda(), rpn_bbox.cuda(), gt_class_ids.cuda(), gt_boxes.cuda(), gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            # print(target_class_ids.size())
            # print(rpn_class_logits.size())
            # print(rpn_pred_bbox.size())
            # print(mrcnn_class_logits.size())
            # print(target_deltas.size())
            # print(mrcnn_bbox.size())
            # print(target_mask.size())
            # print(mrcnn_mask.size())

            # print(target_class_ids)

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = \
                compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            if (batch_count % self.config.BATCH_SIZE) == 0:
                optimizer.step()
                optimizer.zero_grad()
                batch_count = 0

            print_log(', loss: {:.2f}, rpn_closs: {:.2f}, rpn_bloss: {:.2f}, mrcnn_closs: {:.2f}, mrcnn_bloss: {:.2f}, mrcnn_mloss: {:.2f}'.format(
                loss.item(), rpn_class_loss.item(), rpn_bbox_loss.item(), mrcnn_class_loss.item(), mrcnn_bbox_loss.item(), mrcnn_mask_loss.item()), log=self.log_file)

            loss_sum += loss.item()/num_steps
            loss_rpn_class_sum += rpn_class_loss.item()/num_steps
            loss_rpn_bbox_sum += rpn_bbox_loss.item()/num_steps
            loss_mrcnn_class_sum += mrcnn_class_loss.item()/num_steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.item()/num_steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.item()/num_steps

            # Break after 'num_steps' num_steps
            if step==num_steps-1: break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum

    def valid_epoch(self, datagenerator, num_steps):
        step = 0
        loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum = 0, 0, 0, 0, 0, 0
        for images, image_metas, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks, image_index, filename in datagenerator:            
            if islist(gt_class_ids): continue           # TODO, why this happen
            printProgressBar(step + 1, num_steps, log=self.log_file, prefix="\t{}/{}".format(step + 1, num_steps), suffix='index: {:5d}, filename: {:40s}'.format(image_index.item(), filename[0]), length=10)

            image_metas = image_metas.numpy()
            images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = Variable(images, volatile=True), Variable(rpn_match, volatile=True), Variable(rpn_bbox, volatile=True), Variable(gt_class_ids, volatile=True), Variable(gt_boxes, volatile=True), Variable(gt_masks, volatile=True)
            if self.config.GPU_COUNT: images, rpn_match, rpn_bbox, gt_class_ids, gt_boxes, gt_masks = images.cuda(), rpn_match.cuda(), rpn_bbox.cuda(), gt_class_ids.cuda(), gt_boxes.cuda(), gt_masks.cuda()

            # Run object detection
            rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask = \
                self.predict([images, image_metas, gt_class_ids, gt_boxes, gt_masks], mode='training')

            if not target_class_ids.size(): continue

            # Compute losses
            rpn_class_loss, rpn_bbox_loss, mrcnn_class_loss, mrcnn_bbox_loss, mrcnn_mask_loss = compute_losses(rpn_match, rpn_bbox, rpn_class_logits, rpn_pred_bbox, target_class_ids, mrcnn_class_logits, target_deltas, mrcnn_bbox, target_mask, mrcnn_mask)
            loss = rpn_class_loss + rpn_bbox_loss + mrcnn_class_loss + mrcnn_bbox_loss + mrcnn_mask_loss

            print_log(', loss: {:.2f}, rpn_closs: {:.2f}, rpn_bloss: {:.2f}, mrcnn_closs: {:.2f}, mrcnn_bloss: {:.2f}, mrcnn_mloss: {:.2f}'.format(
                loss.item(), rpn_class_loss.item(), rpn_bbox_loss.item(), mrcnn_class_loss.item(), mrcnn_bbox_loss.item(), mrcnn_mask_loss.item()), log=self.log_file)

            # Statistics
            loss_sum += loss.item()/num_steps
            loss_rpn_class_sum += rpn_class_loss.item()/num_steps
            loss_rpn_bbox_sum += rpn_bbox_loss.item()/num_steps
            loss_mrcnn_class_sum += mrcnn_class_loss.item()/num_steps
            loss_mrcnn_bbox_sum += mrcnn_bbox_loss.data.item()/num_steps
            loss_mrcnn_mask_sum += mrcnn_mask_loss.data.item()/num_steps

            # Break after 'num_steps' num_steps
            if step==num_steps-1: break
            step += 1

        return loss_sum, loss_rpn_class_sum, loss_rpn_bbox_sum, loss_mrcnn_class_sum, loss_mrcnn_bbox_sum, loss_mrcnn_mask_sum

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images, image_metas, windows = [], [], []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = resize_image(image, min_dim=self.config.IMAGE_MIN_DIM, max_dim=self.config.IMAGE_MAX_DIM, padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            image_meta = compose_image_meta(0, image.shape, window, np.zeros([self.config.NUM_CLASSES], dtype=np.int32))		            # Build image_meta

            molded_images.append(molded_image)		# Append
            windows.append(window)
            image_metas.append(image_meta)

        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) if full_masks else np.empty((0,) + masks.shape[1:3])

        return boxes, class_ids, scores, full_masks