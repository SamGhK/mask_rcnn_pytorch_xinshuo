# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import argparse, os, torch, random, numpy as np
from mylibs import MaskRCNN, cityscape_class_names, CocoConfig, CocoDataset, CityscapeConfig, CityScapeDataset, Config
from xinshuo_miscellaneous import print_log
torch.backends.cudnn.enabled = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Mask R-CNN.')
    parser.add_argument('command',                          metavar="<command>", help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset',    required=True,      type=str, default='cityscapes', help='dataset name')
    parser.add_argument('--data_dir',   required=True,      type=str, default='/path/to/cityscapes', help='Directory of the dataset')
    parser.add_argument('--model',      required=False,     metavar="/path/to/weights.pth", help="Path to weights .pth file or 'coco'")
    parser.add_argument('--save_dir',   required=False,     default='/media/xinshuo/Data/models/mask_rcnn_pytorch', metavar="/path/to/logs/", help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--manualSeed', required=False,     type=int, default=2345, help='seed')
    args = parser.parse_args()

    # Prepare options
    if args.manualSeed is None: args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    assert args.dataset == 'coco' or args.dataset == 'cityscapes', 'wrong dataset name'

    # Configurations
    if args.command == 'train':
        if args.dataset == 'coco': config = CocoConfig()
        else: config = CityscapeConfig()
    else:
        class InferenceConfig(Config):
            # Set batch size to 1 since we'll be running inference on one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NAME = 'evaluate_%s' % args.dataset
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
            if args.dataset == 'coco': NUM_CLASSES = 1 + 80
            else: NUM_CLASSES = 1 + len(cityscape_class_names)

        config = InferenceConfig()

    model = MaskRCNN(config=config, model_dir=args.save_dir)
    if config.GPU_COUNT: model = model.cuda()

    config.display(model.log_file)
    print_log('Seed: %d' % args.manualSeed, model.log_file)
    print_log('Model: %s' % args.model, model.log_file)
    print_log('Dataset: %s' % args.dataset, model.log_file)
    print_log('Save Directory: %s' % args.save_dir, model.log_file)
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
            dataset_train.load_data(args.data_dir, split='train')
        else:
            dataset_train = CityScapeDataset(args.data_dir, split='train', gttype='gtFine')
            dataset_train.load_data()
        dataset_train.prepare()

        # Validation dataset
        if args.dataset == 'coco':
            dataset_val = CocoDataset()
            dataset_val.load_data(args.data_dir, split='val')
        else:
            dataset_val = CityScapeDataset(args.data_dir, split='val', gttype='gtFine')
            dataset_val.load_data()
        dataset_val.prepare()

        print_log("Training network heads", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, num_epochs=10, layers='heads')
        print_log("Fine tune Resnet stage 4 and up", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, num_epochs=30, layers='4+')
        print_log("Fine tune all layers", model.log_file)
        model.train_model(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10, num_epochs=80, layers='all')
    else: print_log("'{}' is not recognized. " "Use 'train' or 'evaluate'".format(args.command), model.log_file)

    model.log_file.close()