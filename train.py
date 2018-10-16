# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import argparse, os
from config import Config
from mylibs import MaskRCNN
from coco import CocoConfig, CocoDataset, evaluate_coco
from xinshuo_miscellaneous import print_log

ROOT_DIR = os.getcwd()      # Root directory of the project
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")      # Path to trained weights file

save_dir = '/media/xinshuo/Data/models/mask_rcnn_pytorch'
# DEFAULT_LOGS_DIR = os.path.join(save_dir, "logs")       # Directory to save logs and model checkpoints, if not provided
DEFAULT_DATASET_YEAR = "2014"

############################################################
#  Training
############################################################
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command", metavar="<command>", help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True, metavar="/path/to/coco/", help='Directory of the MS-COCO dataset')
    parser.add_argument('--year', required=False, default=DEFAULT_DATASET_YEAR, metavar="<year>", help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False, metavar="/path/to/weights.pth", help="Path to weights .pth file or 'coco'")
    parser.add_argument('--save_dir', required=False, default=save_dir, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False, default=500, metavar="<image count>", help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False, default=False, type=bool, metavar="<True|False>", help='Automatically download and unzip MS-COCO files (default=False)')
    args = parser.parse_args()

    # Configurations
    if args.command == "train": config = CocoConfig()
    else:
        class InferenceConfig(Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()

    # Create model
    if args.command == "train": model = MaskRCNN(config=config, model_dir=args.save_dir)
    else: model = MaskRCNN(config=config, model_dir=args.save_dir)
    if config.GPU_COUNT: model = model.cuda()

    config.display(model.log_file)
    print_log('Command: %s' % args.command, model.log_file)
    print_log('Model: %s' % args.model, model.log_file)
    print_log('Dataset: %s' % args.dataset, model.log_file)
    print_log('Year: %s' % args.year, model.log_file)
    print_log('Save Directory: %s' % args.save_dir, model.log_file)
    print_log('Auto Download: %s' % args.download, model.log_file)

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco": model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last": model_path = model.find_last()[1]        # Find last trained weights
        elif args.model.lower() == "imagenet": model_path = config.IMAGENET_MODEL_PATH         # Start from ImageNet trained weights
        else: model_path = args.model
    else: model_path = ""

    # Load weights
    print_log("Loading weights from %s" % model_path, model.log_file)
    model.load_weights(model_path)

    if args.command == "train":
        # Training dataset. Use the training set and 35K from the validation set, as as in the Mask RCNN paper.
        dataset_train = CocoDataset()
        dataset_train.load_data(args.dataset, "train", year=args.year, auto_download=args.download)
        # dataset_train.load_coco(args.dataset, "valminusminival", year=args.year, auto_download=args.download)
        dataset_train.prepare()
        # zxc

        # Validation dataset
        dataset_val = CocoDataset()
        dataset_val.load_data(args.dataset, "val", year=args.year, auto_download=args.download)
        dataset_val.prepare()

        print_log("Training network heads", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=40, layers='heads')
        print_log("Fine tune Resnet stage 4 and up", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=120, layers='4+')
        print_log("Fine tune all layers", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, epochs=160, layers='all')
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = CocoDataset()
        coco = dataset_val.load_data(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
        dataset_val.prepare()
        print_log("Running COCO evaluation on {} images.".format(args.limit), model.log_file)
        evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
        evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else: print_log("'{}' is not recognized. " "Use 'train' or 'evaluate'".format(args.command), model.log_file)

    model.log_file.close()