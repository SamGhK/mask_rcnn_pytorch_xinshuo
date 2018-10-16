# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import argparse, os, torch, random
from config import Config
from mylibs import MaskRCNN
from coco import CocoConfig, CocoDataset, evaluate_coco
from cityscape import CityscapeConfig, CityScapeDataset
from xinshuo_miscellaneous import print_log

torch.backends.cudnn.enabled = True
ROOT_DIR = os.getcwd()      # Root directory of the project
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")      # Path to trained weights file

save_dir = '/media/xinshuo/Data/models/mask_rcnn_pytorch'
# DEFAULT_LOGS_DIR = os.path.join(save_dir, "logs")       # Directory to save logs and model checkpoints, if not provided
DEFAULT_DATASET_YEAR = "2014"
# DATASET = 'coco'
############################################################
#  Training
############################################################
if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command", metavar="<command>", help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=True, type=str, default='cityscapes', help='dataset name')
    parser.add_argument('--data_dir', required=True, type=str, default='/path/to/cityscapes', help='Directory of the dataset')
    parser.add_argument('--year', required=False, default=DEFAULT_DATASET_YEAR, metavar="<year>", help='Year of the MS-COCO dataset (2014 or 2017) (default=2014)')
    parser.add_argument('--model', required=False, metavar="/path/to/weights.pth", help="Path to weights .pth file or 'coco'")
    parser.add_argument('--save_dir', required=False, default=save_dir, metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False, default=500, metavar="<image count>", help='Images to use for evaluation (default=500)')
    parser.add_argument('--download', required=False, default=False, type=bool, metavar="<True|False>", help='Automatically download and unzip MS-COCO files (default=False)')
    parser.add_argument('--manualSeed', required=False, default='1234', help='seed')
    args = parser.parse_args()

    # Prepare options
    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    assert args.dataset == 'coco' or args.dataset == 'cityscapes', 'wrong dataset name'

    # Configurations
    if args.command == 'train':
        if args.dataset == 'coco': config = CocoConfig()
        else: config = CityscapeConfig()
    else:
        class InferenceConfig(Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NAME = 'evaluate_%s' % args.dataset
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            if args.dataset == 'coco': NUM_CLASSES = 1 + 80
            else: NUM_CLASSES = 1 + 35

        config = InferenceConfig()

    # Create model
    model = MaskRCNN(config=config, model_dir=args.save_dir)
    if config.GPU_COUNT: model = model.cuda()

    config.display(model.log_file)
    # print_log('Command: %s' % args.command, model.log_file)
    print_log('Model: %s' % args.model, model.log_file)
    print_log('Dataset: %s' % args.dataset, model.log_file)
    print_log('Save Directory: %s' % args.save_dir, model.log_file)
    
    if args.dataset == 'coco':
        print_log('Year: %s' % args.year, model.log_file)
        print_log('Auto Download: %s' % args.download, model.log_file)

    # Select weights file to load
    if args.model:
        # if args.model.lower() == "coco": model_path = COCO_MODEL_PATH
        if args.model.lower() == 'imagenet': model_path = config.IMAGENET_MODEL_PATH         # Start from ImageNet trained weights
        elif args.model.lower() == 'last': model_path = model.find_last()[1]        # Find last trained weights
        else: model_path = args.model
    else: model_path = ''

    # train and evaluate
    print_log('Loading weights from %s' % model_path, model.log_file)
    model.load_weights(model_path)
    if args.command == 'train':        
        if args.dataset == 'coco':
            dataset_train = CocoDataset()
            dataset_train.load_data(args.data_dir, subset='train', year=args.year, auto_download=args.download)
        else:
            dataset_train = CityScapeDataset(args.data_dir, split='train', gttype='gtFine')
            dataset_train.load_data()
        dataset_train.prepare()

        # Validation dataset
        if args.dataset == 'coco':
            dataset_val = CocoDataset()
            dataset_val.load_data(args.data_dir, subset='val', year=args.year, auto_download=args.download)
        else:
            dataset_val = CityScapeDataset(args.data_dir, split='val', gttype='gtFine')
            dataset_val.load_data()
        dataset_val.prepare()

        print_log("Training network heads", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, num_epochs=40, layers='heads')
        print_log("Fine tune Resnet stage 4 and up", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, num_epochs=120, layers='4+')
        print_log("Fine tune all layers", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, num_epochs=160, layers='all')
    # elif args.command == "evaluate":
    #     # Validation dataset
    #     dataset_val = CocoDataset()
    #     coco = dataset_val.load_data(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
    #     dataset_val.prepare()
    #     print_log("Running COCO evaluation on {} images.".format(args.limit), model.log_file)
    #     evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    #     evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    else: print_log("'{}' is not recognized. " "Use 'train' or 'evaluate'".format(args.command), model.log_file)

    model.log_file.close()