import os
import glob

import torch
from torch.autograd import Variable
from torch.utils import data, model_zoo
from torch.utils.data import Subset

import numpy as np

import cv2
from PIL import Image

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1].absolute().resolve(strict=True)

import cfg as config

from utils.dataset import BasicDataSet
from utils import helper_utils

from model.deeplab import Deeplab
from model.deeplab_depth import DeeplabDepth
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG

from utils.compute_iou import compute_mIoU
from utils.compute_Fwb import weightedFb

#######################################
#######################################

def eval_model(model, test_loader, eval_mIoU=False, eval_Fwb=False, verbose=False):
    """TESTING WITH mIoU"""

    ######################
    # INIT
    ######################

    if not os.path.exists(config.TEST_SAVE_FOLDER):
        os.makedirs(config.TEST_SAVE_FOLDER)

    gt_pred_images = glob.glob(config.TEST_SAVE_FOLDER + '*')
    for images in gt_pred_images:
        os.remove(images)

    ######################
    ######################
    model.eval()

    for image_idx, batch in enumerate(test_loader):

        images, labels, depths = batch['image'], batch['label'], batch['depth']
        images = images.to(device=config.GPU, dtype=torch.float32)
        labels = labels.to(device=config.GPU, dtype=torch.long)
        depths = depths.to(device=config.GPU, dtype=torch.float32)

        if config.MODEL == 'DeepLab' or config.MODEL == 'DeepLabv3':
            # for depth based training
            if config.NUM_CHANNELS == 1:
                pred_test_main = model(depths)
            else:
                pred_test_main = model(images)
        elif config.MODEL == 'DeepLabDepth' or config.MODEL == 'DeepLabv3Depth':
            pred_test_main = model(images, depths)
        elif config.MODEL == 'DeepLabMulti' or config.MODEL == 'DeepLabv3Multi':
            pred_test_aux, pred_test_main = model(images)
        elif config.MODEL == 'DeepLabv3DepthMulti':
            pred_test_aux, pred_test_main = model(images, depths)
        elif config.MODEL == 'DeeplabVGG' or config.MODEL == 'Oracle':
            pred_test_main = model(images)

        images = helper_utils.cuda_2_numpy(images)
        depths = helper_utils.cuda_2_numpy(depths)
        labels = helper_utils.cuda_2_numpy(labels)
        pred_test_main = helper_utils.cuda_2_numpy(pred_test_main, is_pred=True)

        if verbose and image_idx == 0:
            print(f"\n Running Pred on {image_idx} Image, AFF Labes {np.unique(pred_test_main)} ..")

        ##################
        ### SAVE IMGS
        ##################

        gt_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_GT_EXT
        pred_name = config.TEST_SAVE_FOLDER + str(image_idx) + config.TEST_PRED_EXT

        Image.fromarray(labels).save(gt_name)
        Image.fromarray(pred_test_main).save(pred_name)

        ##################
        ### PLOTTING IMGS
        ##################
        if verbose:
            imgs = helper_utils.torch_2_numpy(images, mean=config.IMG_MEAN, is_rgb=True)
            depths = helper_utils.torch_2_numpy(depths, mean=config.IMG_MEAN, is_depth=True)
            labels = helper_utils.torch_2_numpy(labels)
            labels = helper_utils.colorize_mask(labels)
            pred = helper_utils.torch_2_numpy(pred_test_main)
            pred = helper_utils.colorize_mask(pred)

            cv2.imshow('rgb', cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
            cv2.imshow('depth', np.array(depths, dtype=np.uint8))
            cv2.imshow('label', labels)
            cv2.imshow('pred', pred)
            cv2.waitKey(1)

    model.train()

    if eval_mIoU:
        ##################
        ### compute mIoU
        ##################
        mIoU = compute_mIoU(verbose=verbose)
        return mIoU
    elif eval_Fwb:
        ##################
        ### compute Fwb
        ##################
        Fwb = weightedFb(verbose=verbose, visualize=False)
        return Fwb