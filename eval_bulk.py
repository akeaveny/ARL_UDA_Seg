import os
import glob

import torch
from torch.utils import data, model_zoo

######################
######################

import cfg as config

from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG

from utils.dataset import BasicDataSet

from utils.eval_metrics import eval_model

###############################
###############################

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

###############################
###############################

def eval(model, test_loader):
    mIoU = eval_mIoU(model=model, test_loader=test_loader, verbose=False)
    print("mIoU: ", mIoU)

###############################
###############################

def main():

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
    test_loader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    ######################
    ######################

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

    trained_models = sorted(glob.glob(config.RESTORE_TRAINED_FOLDER))
    for trained_model in trained_models:

        model.load_state_dict(torch.load(trained_model))
        print("restoring trained weights .. {}".format(trained_model))

        model.eval()
        model.cuda(gpu)

        eval(model, test_loader)

if __name__ == "__main__":
    main()
