import glob

import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from torch.utils.data import Subset
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import cv2
import matplotlib.pyplot as plt
import torch.nn as nn

import config

######################
######################

import config
from utils.loss import CrossEntropy2d, lr_poly, adjust_learning_rate, adjust_learning_rate_D
from utils import helper_utils
from utils.dataset import BasicDataSet

#######################################
#######################################

def main():
    """Create the model and start the evaluation process."""

    if not os.path.exists(config.TEST_SAVE_PATH):
        os.makedirs(config.TEST_SAVE_PATH)

    pred_gt_pred_images = glob.glob(config.TEST_SAVE_PATH + '*')
    for image in pred_gt_pred_images:
        os.remove(image)

    ######################
    # LOAD MODEL
    ######################

    gpu = config.GPU

    if config.MODEL == 'DeepLabMulti':
        model = DeeplabMulti(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 'Oracle':
        model = Res_Deeplab(num_classes=config.NUM_CLASSES)
    elif config.MODEL == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=config.NUM_CLASSES)
    else:
        assert "*** No Model Selected ***"

    if config.RESTORE_TRAINED_MODEL[:4] == 'http':
        saved_state_dict = model_zoo.load_url(config.RESTORE_TRAINED_MODEL)
        ### for running different versions of pytorch
        model_dict = model.state_dict()
        saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
        model_dict.update(saved_state_dict)
        ###
        model.load_state_dict(saved_state_dict)
    else:
        model.load_state_dict(torch.load(config.RESTORE_TRAINED_MODEL))

    model.eval()
    model.cuda(gpu)

    ######################
    # LOADING CITYSCAPE
    ######################

    dataset = BasicDataSet(config.DATA_DIRECTORY_TARGET_VAL,
                           gta5_remap_label_idx=False,
                           ### EXTENDING DATASET
                           extend_dataset=False,
                           ### PRE-PROCESSING
                           mean=config.IMG_MEAN, crop_size=config.INPUT_SIZE_TARGET, ignore_label=255,
                           ### IMGAUG
                           apply_imgaug=False)

    testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    ######################
    ######################

    for index, batch in enumerate(testloader):
        print('%d processd' % index)
        # image, label = batch
        image, label = batch['image'], batch['label']
        image = Variable(image, volatile=True).cuda(gpu)
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

        # helper_utils.print_class_labels(seg_name='GT', seg_mask=label)
        # helper_utils.print_class_labels(seg_name='PRED', seg_mask=output)

        gt_name = config.TEST_SAVE_PATH + str(index) + config.TEST_GT_EXT
        pred_name = config.TEST_SAVE_PATH + str(index) + config.TEST_PRED_EXT

        Image.fromarray(label).save(gt_name)
        Image.fromarray(output).save(pred_name)

        ##################
        ### plotting
        ##################

        cv2.imshow("image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.imshow("gt", label * 10)
        cv2.imshow("pred", output * 10)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
