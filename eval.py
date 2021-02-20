import os

import numpy as np

import torch
from torch.utils import data, model_zoo
from torch.utils.data import Subset

######################
######################

import cfg as config

from model.deeplab import Deeplab
from model.deeplab_depth import DeeplabDepth
from model.deeplab_multi import DeeplabMulti
from model.deeplabv3 import DeepLabv3
from model.deeplabv3_depth import DeepLabv3Depth
from model.deeplab_vgg import DeeplabVGG

from utils.dataset import BasicDataSet

from utils.eval_metrics import eval_model

###############################
###############################

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

###############################
###############################

def eval(model, test_loader):
    # mIoU = eval_model(model=model, test_loader=test_loader, eval_mIoU=True, verbose=True)
    # print("mIoU: ", mIoU)
    Fwb = eval_model(model=model, test_loader=test_loader, eval_Fwb=True, verbose=True)
    print("Fwb: ", Fwb)

###############################
###############################

def main():

    ######################
    # LOADING CITYSCAPE
    ######################

    dataset = BasicDataSet(
                           ### PRE-PROCESSING
                           config.DATA_DIRECTORY_TARGET_TEST,
                           resize=config.RESIZE_TARGET,
                           mean=config.IMG_MEAN_TARGET,
                           std=config.IMG_STD_TARGET,
                           crop_size=config.INPUT_SIZE_TARGET,
                           ### DEPTH
                           use_depth_imgs=config.USE_DEPTH_IMGS,
                           ### MASK
                           gta5_remap_label_idx=False,
                           ignore_label=config.IGNORE_LABEL,
                           ### EXTENDING DATASET
                           extend_dataset=False,
                           max_iters=config.NUM_STEPS,
                           ### IMGAUG
                           apply_imgaug=False)

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(config.RANDOM_SEED)
    total_idx = np.arange(0, len(dataset), 1)
    val_idx = np.random.choice(total_idx, size=int(100), replace=False)
    dataset = Subset(dataset, val_idx)

    test_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    print(f'Test Dataset has {len(test_loader)} examples ..')

    ######################
    # LOAD MODEL
    ######################

    gpu = config.GPU

    if config.MODEL == 'DeepLab':
        model = Deeplab(pretrained=False)
    if config.MODEL == 'DeepLabDepth':
        model = DeeplabDepth(pretrained=False)
    elif config.MODEL == 'DeepLabMulti':
        model = DeeplabMulti(pretrained=False)
    elif config.MODEL == 'DeepLabv3':
        model = DeepLabv3(pretrained=False)
    elif config.MODEL == 'DeepLabv3Depth':
        model = DeepLabv3Depth(pretrained=False)
    else:
        assert "*** No Model Selected ***"

    if config.RESTORE_TRAINED_MODEL[:4] == 'http':
        model.load_state_dict(model_zoo.load_url(config.RESTORE_TRAINED_MODEL))
    else:
        model.load_state_dict(torch.load(config.RESTORE_TRAINED_MODEL))
    print("restoring trained model .. {}".format(config.RESTORE_TRAINED_MODEL))

    model.eval()
    model.cuda(gpu)

    ######################
    ######################

    eval(model, test_loader)

if __name__ == "__main__":
    main()
