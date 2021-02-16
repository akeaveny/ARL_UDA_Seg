import os
import sys

import random
import numpy as np

import torch
from torch.utils import data, model_zoo
from torch.utils.data import DataLoader, random_split, Subset
import torch.backends.cudnn as cudnn

###############################
###############################

import cfg as config

from model.deeplab import Deeplab
from model.deeplab_depth import DeeplabDepth
from model.deeplab_multi import DeeplabMulti
from model.deeplabv3 import DeepLabv3
from model.deeplabv3_depth import DeepLabv3Depth
from model.deeplabv3_multi import DeepLabv3Multi
from model.deeplabv3_depth_multi import DeepLabv3DepthMulti
from model.discriminator import FCDiscriminator

from utils.dataset import BasicDataSet

from torch.utils.tensorboard import SummaryWriter

from scripts.train_segmentation import train_segmentation
from scripts.train_segmentation_multi import train_segmentation_multi
from scripts.train_AdaptSegNet_multi import train_AdaptSegNet_multi
from scripts.train_CLAN_multi import train_CLAN_multi

###############################
###############################

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

###############################
###############################

def main():
    """Create the model and start the training."""
    print('saving to .. {}'.format(config.SNAPSHOT_DIR))

    ######################
    # INIT
    ######################

    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

    ######################
    # LOAD MODEL
    ######################

    # Create network
    if config.MODEL == 'DeepLab':
        model = Deeplab(pretrained=config.LOAD_PRETRAINED_WEIGHTS)
    elif config.MODEL == 'DeepLabDepth':
        model = DeeplabDepth(pretrained=config.LOAD_PRETRAINED_WEIGHTS)
    elif config.MODEL == 'DeepLabMulti':
        model = DeeplabMulti(pretrained=config.LOAD_PRETRAINED_WEIGHTS)
    elif config.MODEL == 'DeepLabv3':
        model = DeepLabv3(pretrained=config.LOAD_PRETRAINED_WEIGHTS)
    elif config.MODEL == 'DeepLabv3Depth':
        model = DeepLabv3Depth(pretrained=config.LOAD_PRETRAINED_WEIGHTS)
    elif config.MODEL == 'DeepLabv3Multi':
        model = DeepLabv3Multi(pretrained=config.LOAD_PRETRAINED_WEIGHTS)
    elif config.MODEL == 'DeepLabv3DepthMulti':
        model = DeepLabv3DepthMulti(pretrained=config.LOAD_PRETRAINED_WEIGHTS)

    # restore checkpoint
    if config.RESTORE_CHECKPOINT is not None:
        if config.RESTORE_CHECKPOINT[:4] == 'http':
            model.load_state_dict(model_zoo.load_url(config.RESTORE_CHECKPOINT))
        else:
            model.load_state_dict(torch.load(config.RESTORE_CHECKPOINT))

    model.train()
    model.cuda(config.GPU)
    cudnn.benchmark = True
    cudnn.enabled = True

    ######################
    # LOAD DIS
    ######################

    # init D
    model_D1 = FCDiscriminator(num_classes=config.NUM_CLASSES)
    model_D2 = FCDiscriminator(num_classes=config.NUM_CLASSES)

    model_D1.train()
    model_D2.train()

    model_D1.cuda(config.GPU)
    model_D2.cuda(config.GPU)

    ######################
    # LOGGING
    ######################

    if not os.path.exists(config.SNAPSHOT_DIR):
        os.makedirs(config.SNAPSHOT_DIR)

    ### TENSORBOARD
    writer = SummaryWriter(f'{config.SNAPSHOT_DIR}')

    ######################
    # LOADING SOURCE
    ######################
    if config.FRAMEWORK == 'AdaptSegNet' or config.FRAMEWORK == 'CLAN':
        print("loading source ..")

        source_dataset = BasicDataSet(
                               config.DATA_DIRECTORY_SOURCE_TRAIN,
                               gta5_remap_label_idx=False,
                               ### EXTENDING DATASET
                               extend_dataset=True,
                               max_iters=config.NUM_STEPS,
                               ### PRE-PROCESSING
                               mean=config.IMG_MEAN, crop_size=config.INPUT_SIZE_TARGET, ignore_label=config.IGNORE_LABEL,
                               ### IMGAUG
                               apply_imgaug=True,
                               ### DEPTH
                               use_depth_imgs=config.USE_DEPTH_IMGS)
        assert (len(source_dataset) >= config.NUM_STEPS)

        source_loader = enumerate(DataLoader(source_dataset,
                                           batch_size=config.BATCH_SIZE,
                                           shuffle=True,
                                           num_workers=config.NUM_WORKERS,
                                           pin_memory=True))

    ######################
    # LOADING TARGET
    ######################
    print("loading target ..")

    target_dataset = BasicDataSet(
                           # dataset_dir=config.DATA_DIRECTORY_SOURCE_TRAIN,        # SYN
                           dataset_dir=config.DATA_DIRECTORY_TARGET_TRAIN,      # REAL
                           gta5_remap_label_idx=config.REMAP_LABEL,
                           ### EXTENDING DATASET
                           extend_dataset=True,
                           max_iters=config.NUM_STEPS,
                           ### PRE-PROCESSING
                           mean=config.IMG_MEAN, crop_size=config.INPUT_SIZE, ignore_label=config.IGNORE_LABEL,
                           ### IMGAUG
                           apply_imgaug=True,
                           ### DEPTH
                           use_depth_imgs=config.USE_DEPTH_IMGS)
    assert (len(target_dataset) >= config.NUM_STEPS)

    target_loader = enumerate(DataLoader(target_dataset,
                                         batch_size=config.BATCH_SIZE,
                                         shuffle=True,
                                         num_workers=config.NUM_WORKERS,
                                         pin_memory=True))

    ######################
    # LOADING TEST
    ######################
    print("loading val ..")

    dataset = BasicDataSet(config.DATA_DIRECTORY_TARGET_VAL,
                           gta5_remap_label_idx=False,
                           ### EXTENDING DATASET
                           extend_dataset=False,
                           ### PRE-PROCESSING
                           mean=config.IMG_MEAN, crop_size=config.INPUT_SIZE_TARGET, ignore_label=255,
                           ### IMGAUG
                           apply_imgaug=False,
                           ### DEPTH
                           use_depth_imgs=config.USE_DEPTH_IMGS)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(dataset), 1)
    val_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    dataset = Subset(dataset, val_idx)

    test_loader = data.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True)
                                  # drop_last=True

    ######################
    # UDA TRAINING
    ######################
    try:
        if config.FRAMEWORK == 'Segmentation':
            train_segmentation(model,
                                   target_loader=target_loader,
                                   test_loader=test_loader,
                                   writer=writer)
        if config.FRAMEWORK == 'SegmentationMulti':
            train_segmentation_multi(model,
                                     target_loader=target_loader,
                                     test_loader=test_loader,
                                     writer=writer)
        elif config.FRAMEWORK == 'AdaptSegNet':
            train_AdaptSegNet_multi(model, model_D1, model_D2,
                                       target_loader=target_loader, source_loader=source_loader,
                                       test_loader=test_loader,
                                       writer=writer)
        elif config.FRAMEWORK == 'CLAN':
            train_CLAN_multi(model, model_D1,
                                 target_loader=target_loader, source_loader=source_loader,
                                 test_loader=test_loader,
                                 writer=writer)
    except KeyboardInterrupt:
        print('Saved interrupt..')
        torch.save(model.state_dict(),    config.MODEL_SAVE_PATH + "SEG_INTERRUPTED.pth")
        torch.save(model_D1.state_dict(), config.MODEL_SAVE_PATH + "DIS1_INTERRUPTED.pth")
        torch.save(model_D2.state_dict(), config.MODEL_SAVE_PATH + "DIS2_INTERRUPTED.pth")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    main()
