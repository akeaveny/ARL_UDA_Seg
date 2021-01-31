import argparse
import torchvision
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import random

from model.deeplab_multi import DeeplabMulti

from torch.utils.data import DataLoader, random_split, Subset

###############################
###############################

import config

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from evaluate_mIoU import eval

from utils.loss import CrossEntropy2d, adjust_learning_rate
from utils import helper_utils
from utils.dataset import BasicDataSet

###############################
###############################

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

###############################
###############################

def main():
    """Create the model and start the training."""
    print('saving to .. {}'.format(config.SNAPSHOT_DIR))

    ######################
    # LOAD MODEL
    ######################

    cudnn.enabled = True
    gpu = config.GPU

    # Create network
    if config.MODEL == 'DeepLabMulti':
        model = DeeplabMulti(num_classes=config.NUM_CLASSES, pretrained=config.LOAD_PRETRAINED_WEIGHTS)
        if config.RESTORE_CHECKPOINT is not None:
            model.load_state_dict(torch.load(config.RESTORE_CHECKPOINT))

    model.train()
    model.cuda(config.GPU)
    cudnn.benchmark = True

    ######################
    # LOGGING
    ######################

    if not os.path.exists(config.SNAPSHOT_DIR):
        os.makedirs(config.SNAPSHOT_DIR)

    ### TENSORBOARD
    writer = SummaryWriter(f'{config.SNAPSHOT_DIR}')

    ######################
    # LOADING GTA5
    ######################

    dataset = BasicDataSet(config.DATA_DIRECTORY_SOURCE_TRAIN,
                           gta5_remap_label_idx=True,
                           ### EXTENDING DATASET
                           extend_dataset=config.EXTEND_DATASET, max_iters=config.NUM_STEPS * config.ITER_SIZE * config.BATCH_SIZE,
                           ### PRE-PROCESSING
                           mean=config.IMG_MEAN, crop_size=config.INPUT_SIZE, ignore_label=255,
                           ### IMGAUG
                           apply_imgaug=True)
    assert (len(dataset) >= config.NUM_STEPS)

    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    train_loader_iter = enumerate(train_loader)

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

    ### SELECTING A SUBSET OF IMAGES
    np.random.seed(0)
    total_idx = np.arange(0, len(dataset), 1)
    val_idx = np.random.choice(total_idx, size=int(config.NUM_TEST), replace=False)
    dataset = Subset(dataset, val_idx)

    testloader = data.DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    ######################
    # LOSS
    ######################

    criterion = CrossEntropy2d().cuda(gpu)

    ######################
    # OPTIMIZER
    ######################

    optimizer = optim.SGD(model.optim_parameters(config.LEARNING_RATE),
                          lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    optimizer.zero_grad()

    ######################
    ######################

    best_mIoU = -np.inf
    i_iter = config.StartIteration
    with tqdm(total=config.NUM_STEPS-config.StartIteration, desc=f'Iterations {config.NUM_STEPS}', unit='images') as pbar:
        while i_iter < config.NUM_STEPS:

            loss_seg_value1 = 0
            loss_seg_value2 = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

            for sub_i in range(config.ITER_SIZE):

                _, batch = train_loader_iter.__next__()
                images, labels = batch['image'], batch['label']
                images = images.to(device=config.GPU, dtype=torch.float32)
                labels = labels.to(device=config.GPU, dtype=torch.long)

                pred1, pred2 = model(images)

                loss_seg1 = criterion(pred1, labels)
                loss_seg2 = criterion(pred2, labels)
                loss = loss_seg2 + config.LAMBDA_SEG * loss_seg1

                # proper normalization
                loss = loss / config.ITER_SIZE
                loss.backward()
                loss_seg_value1 += loss_seg1.data.cpu().numpy() / config.ITER_SIZE
                loss_seg_value2 += loss_seg2.data.cpu().numpy() / config.ITER_SIZE

                ### TENSORBOARD
                writer.add_scalar('Loss/train', loss_seg_value1 + loss_seg_value2, i_iter)
                writer.add_scalar('Loss/seg_loss1', loss_seg_value1, i_iter)
                writer.add_scalar('Loss/seg_loss2', loss_seg_value2, i_iter)

            optimizer.step()

            pbar.set_postfix(**{
                                'SegLoss: ': loss_seg_value1 + loss_seg_value2,
                                'SegLoss1: ': loss_seg_value1,
                                'SegLoss2: ': loss_seg_value2})

            i_iter += 1
            pbar.update(images.shape[0])

            ## EVAL
            if i_iter != 0 and i_iter % config.EVAL_UPDATE == 0:
                mIoU = eval(model, testloader)
                writer.add_scalar('eval/mIoU', mIoU, i_iter)
                if mIoU > best_mIoU:
                    best_mIoU = mIoU
                    print("Saving best model .. best mIoU={:.5} ..".format(best_mIoU))
                    writer.add_scalar('eval/Best mIoU', best_mIoU, i_iter)
                    torch.save(model.state_dict(), config.BEST_MODEL_SAVE_PATH)
                    ### TENSORBOARD: loading best images
                    writer.add_images('source/gt_mask',
                                      helper_utils.cuda_label_2_tensorboard(labels),
                                      i_iter)
                    writer.add_images('source/pred_mask',
                                      helper_utils.cuda_label_2_tensorboard(pred2, is_pred=True),
                                      i_iter)

            ## TENSORBOARD
            writer.add_scalar('learning_rate/seg', optimizer.param_groups[0]['lr'], i_iter)
            # DEEPLAB WEIGHTS AND GRADS
            if i_iter != 0 and i_iter % config.TENSORBOARD_UPDATE == 0:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        # print('Seg_Layer: ', tag.split('/'))
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), i_iter)
                        pass
                    else:
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), i_iter)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), i_iter)

    if i_iter >= config.NUM_STEPS - 1:
        print("Saving final model, mIoU={:.5} ..".format(mIoU))
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH + "final_saved_model_{:.5}_mIoU.pth".format(mIoU))

if __name__ == '__main__':
    main()
