import glob
import sys

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from torch.utils.data import Subset

import os
from PIL import Image

import cv2
import numpy as np

######################
######################

import config
from utils import helper_utils
from utils.dataset import BasicDataSet
from compute_iou import compute_mIoU

#######################################
#######################################

def eval(model, testloader):
    """Create the model and start the evaluation process."""

    if not os.path.exists(config.TEST_SAVE_PATH):
        os.makedirs(config.TEST_SAVE_PATH)

    pred_gt_pred_images = glob.glob(config.TEST_SAVE_PATH + '*')
    for image in pred_gt_pred_images:
        os.remove(image)

    ######################
    ######################
    model.eval()

    for index, batch in enumerate(testloader):

        image, label = batch['image'], batch['label']
        image = Variable(image, volatile=True).cuda(config.GPU)

        if config.MODEL == 'DeepLabMulti':
            output1, output2 = model(image)
            output = output2
        elif config.MODEL == 'DeeplabVGG' or config.MODEL == 'Oracle':
            output = model(image)

        image = helper_utils.cuda_2_numpy(image, is_rgb=True)
        label = helper_utils.cuda_2_numpy(label)
        output = helper_utils.cuda_2_numpy(output, is_pred=True)

        ##################
        ### SAVE IMGS
        ##################

        gt_name = config.TEST_SAVE_PATH + str(index) + config.TEST_GT_EXT
        pred_name = config.TEST_SAVE_PATH + str(index) + config.TEST_PRED_EXT

        Image.fromarray(label).save(gt_name)
        Image.fromarray(output).save(pred_name)

    model.train()
    ##################
    ### compute mIoU
    ##################
    mIoU = compute_mIoU(verbose=False)
    # print("mIoU: ", mIoU)
    return mIoU

if __name__ == '__main__':

    ######################
    # LOAD MODEL
    ######################

    gpu = config.GPU

    if config.MODEL == 'DeepLab':
        model = DeeplabMulti(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 'Oracle':
        model = Res_Deeplab(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=config.NUM_CLASSES)
    else:
        assert "*** No Model Selected ***"

    model.load_state_dict(torch.load(config.RESTORE_TRAINED_MODEL))

    model.eval()
    model.cuda(gpu)

    ######################
    # LOADING CITYSCAPE
    ######################

    w, h = map(int, config.INPUT_SIZE_TARGET.split(','))
    input_size = (w, h)

    dataset = BasicDataSet(config.DATA_DIRECTORY_TARGET_VAL,
                           gta5_remap_label_idx=False,
                           ### EXTENDING DATASET
                           extend_dataset=False,
                           ### PRE-PROCESSING
                           mean=config.IMG_MEAN, crop_size=input_size, ignore_label=255,
                           ### IMGAUG
                           apply_imgaug=False)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(0)
    total_idx = np.arange(0, len(dataset), 1)
    val_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    dataset = Subset(dataset, val_idx)

    testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    eval(model=model, testloader=testloader)
