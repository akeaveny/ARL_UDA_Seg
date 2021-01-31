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
from model.discriminator import FCDiscriminator

from torch.utils.data import DataLoader, random_split, Subset

###############################
###############################

import config

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from evaluate_mIoU import eval

from utils.loss import CrossEntropy2d, adjust_learning_rate, adjust_learning_rate_D
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
    # LOADING GTA5
    ######################

    dataset = BasicDataSet(config.DATA_DIRECTORY_SOURCE_TRAIN,
                           gta5_remap_label_idx=True,
                           ### EXTENDING DATASET
                           extend_dataset=config.EXTEND_DATASET,
                           max_iters=config.NUM_STEPS * config.ITER_SIZE * config.BATCH_SIZE,
                           ### PRE-PROCESSING
                           mean=config.IMG_MEAN, crop_size=config.INPUT_SIZE, ignore_label=255,
                           ### IMGAUG
                           apply_imgaug=True)
    assert (len(dataset) >= config.NUM_STEPS)

    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    train_loader_iter = enumerate(train_loader)

    ######################
    # LOADING CITYSCAPES
    ######################

    dataset = BasicDataSet(config.DATA_DIRECTORY_TARGET_TRAIN,
                           gta5_remap_label_idx=False,
                           ### EXTENDING DATASET
                           extend_dataset=config.EXTEND_DATASET,
                           max_iters=config.NUM_STEPS * config.ITER_SIZE * config.BATCH_SIZE,
                           ### PRE-PROCESSING
                           mean=config.IMG_MEAN, crop_size=config.INPUT_SIZE_TARGET, ignore_label=255,
                           ### IMGAUG
                           apply_imgaug=True)
    assert (len(dataset) >= config.NUM_STEPS)

    target_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    target_loader_iter = enumerate(target_loader)

    ######################
    # LOADING VAL
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

    if config.GAN == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif config.GAN == 'LS':
        bce_loss = torch.nn.MSELoss()

    ######################
    # OPTIMIZER
    ######################

    optimizer = optim.SGD(model.optim_parameters(config.LEARNING_RATE),
                          lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=config.LEARNING_RATE_D, betas=(0.9, 0.99))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=config.LEARNING_RATE_D, betas=(0.9, 0.99))

    optimizer.zero_grad()
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()

    ######################
    ######################

    # labels for adversarial training
    source_label = 0
    target_label = 1

    best_mIoU = -np.inf
    i_iter = config.StartIteration
    with tqdm(total=config.NUM_STEPS - config.StartIteration, desc=f'Iterations {config.NUM_STEPS}',
              unit='images') as pbar:
        while i_iter < config.NUM_STEPS:

            loss_seg_value1 = 0
            loss_adv_target_value1 = 0
            loss_D_value1 = 0

            loss_seg_value2 = 0
            loss_adv_target_value2 = 0
            loss_D_value2 = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            adjust_learning_rate_D(optimizer_D1, i_iter)
            adjust_learning_rate_D(optimizer_D2, i_iter)

            for sub_i in range(config.ITER_SIZE):

                ######################
                # train G
                ######################

                # don't accumulate grads in D
                for param in model_D1.parameters():
                    param.requires_grad = False

                for param in model_D2.parameters():
                    param.requires_grad = False

                # train with source

                _, batch = train_loader_iter.__next__()
                images, labels = batch['image'], batch['label']
                images = images.to(device=config.GPU, dtype=torch.float32)
                labels = labels.to(device=config.GPU, dtype=torch.long)

                pred1, pred2 = model(images)
                source_gt, source_pred = labels, pred2

                loss_seg1 = criterion(pred1, labels)
                loss_seg2 = criterion(pred2, labels)
                loss = loss_seg2 + config.LAMBDA_SEG * loss_seg1

                # proper normalization
                loss = loss / config.ITER_SIZE
                loss.backward()
                loss_seg_value1 += loss_seg1.data.cpu().numpy() / config.ITER_SIZE
                loss_seg_value2 += loss_seg2.data.cpu().numpy() / config.ITER_SIZE

                ### TENSORBOARD
                writer.add_scalar('SegLoss/SegLoss', loss_seg_value1 + loss_seg_value2, i_iter)
                writer.add_scalar('SegLoss/loss_seg_value1', loss_seg_value1, i_iter)
                writer.add_scalar('SegLoss/loss_seg_value2', loss_seg_value2, i_iter)

                # train with target

                _, batch = target_loader_iter.__next__()
                images, labels = batch['image'], batch['label']
                images = images.to(device=config.GPU, dtype=torch.float32)
                labels = labels.to(device=config.GPU, dtype=torch.long)

                pred_target1, pred_target2 = model(images)
                target_gt, target_pred = labels, pred_target2

                D_out1 = model_D1(F.softmax(pred_target1))
                D_out2 = model_D2(F.softmax(pred_target2))

                loss_adv_target1 = bce_loss(D_out1,
                                           Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(
                                               config.GPU))

                loss_adv_target2 = bce_loss(D_out2,
                                            Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(
                                                config.GPU))

                loss = config.LAMBDA_ADV_TARGET1 * loss_adv_target1 + config.LAMBDA_ADV_TARGET2 * loss_adv_target2
                loss = loss / config.ITER_SIZE
                loss.backward()
                loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy() / config.ITER_SIZE
                loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy() / config.ITER_SIZE

                ### TENSORBOARD
                writer.add_scalar('ADVLoss/ADVLoss', loss_adv_target_value1 + loss_adv_target_value2, i_iter)
                writer.add_scalar('ADVLoss/seg_adv_loss1', loss_adv_target_value1, i_iter)
                writer.add_scalar('ADVLoss/seg_adv_loss2', loss_adv_target_value2, i_iter)

                ######################
                # train G
                ######################

                # bring back requires_grad
                for param in model_D1.parameters():
                    param.requires_grad = True

                for param in model_D2.parameters():
                    param.requires_grad = True

                # train with source
                pred1 = pred1.detach()
                pred2 = pred2.detach()

                D_out1 = model_D1(F.softmax(pred1))
                D_out2 = model_D2(F.softmax(pred2))

                loss_D1 = bce_loss(D_out1,
                                  Variable(torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda(config.GPU))

                loss_D2 = bce_loss(D_out2,
                                   Variable(torch.FloatTensor(D_out2.data.size()).fill_(source_label)).cuda(config.GPU))

                loss_D1 = loss_D1 / config.ITER_SIZE / 2
                loss_D2 = loss_D2 / config.ITER_SIZE / 2

                loss_D1.backward()
                loss_D2.backward()

                loss_D_value1 += loss_D1.data.cpu().numpy()
                loss_D_value2 += loss_D2.data.cpu().numpy()

                # train with target
                pred_target1 = pred_target1.detach()
                pred_target2 = pred_target2.detach()

                D_out1 = model_D1(F.softmax(pred_target1))
                D_out2 = model_D2(F.softmax(pred_target2))

                loss_D1 = bce_loss(D_out1,
                                  Variable(torch.FloatTensor(D_out1.data.size()).fill_(target_label)).cuda(config.GPU))

                loss_D2 = bce_loss(D_out2,
                                   Variable(torch.FloatTensor(D_out2.data.size()).fill_(target_label)).cuda(config.GPU))

                loss_D1 = loss_D1 / config.ITER_SIZE / 2
                loss_D2 = loss_D2 / config.ITER_SIZE / 2

                loss_D1.backward()
                loss_D2.backward()

                loss_D_value1 += loss_D1.data.cpu().numpy()
                loss_D_value2 += loss_D2.data.cpu().numpy()

                ## TENSORBOARD
                writer.add_scalar('DisLoss/DisLoss', loss_D_value1 + loss_D_value2, i_iter)
                writer.add_scalar('DisLoss/loss_D_value1', loss_D_value1, i_iter)
                writer.add_scalar('DisLoss/loss_D_value2', loss_D_value2, i_iter)

            optimizer.step()
            optimizer_D1.step()
            optimizer_D2.step()

            pbar.set_postfix(**{
                'SegLoss: ': loss_seg_value1 + loss_seg_value2,
                'ADVLoss: ': loss_adv_target_value1 + loss_adv_target_value2,
                'DisLoss: ': loss_D_value1 + loss_D_value2})

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
                    torch.save(model_D1.state_dict(), config.BEST_DIS1_SAVE_PATH)
                    torch.save(model_D2.state_dict(), config.BEST_DIS2_SAVE_PATH)
                    ### TENSORBOARD: loading best images
                    writer.add_images('source/gt_mask',
                                      helper_utils.cuda_label_2_tensorboard(source_gt),
                                      i_iter)
                    writer.add_images('source/pred_mask',
                                      helper_utils.cuda_label_2_tensorboard(source_pred, is_pred=True),
                                      i_iter)
                    writer.add_images('target/gt_mask',
                                      helper_utils.cuda_label_2_tensorboard(target_gt),
                                      i_iter)
                    writer.add_images('target/pred_mask',
                                      helper_utils.cuda_label_2_tensorboard(target_pred, is_pred=True),
                                      i_iter)

            ## TENSORBOARD
            writer.add_scalar('learning_rate/seg', optimizer.param_groups[0]['lr'], i_iter)

            if i_iter != 0 and i_iter % config.TENSORBOARD_UPDATE == 0:

                # DEEPLAB WEIGHTS AND GRADS
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        # print('Seg_Layer: ', tag.split('/'))
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), i_iter)
                        pass
                    else:
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), i_iter)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), i_iter)
                # Discriminator 1
                for tag, value in model_D1.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        # print('Dis1_Layer: ', tag.split('/'))
                        writer.add_histogram('dis_weights1/' + tag, value.data.cpu().numpy(), i_iter)
                        pass
                    else:
                        writer.add_histogram('dis_weights1/' + tag, value.data.cpu().numpy(), i_iter)
                        writer.add_histogram('dis_grads1/' + tag, value.grad.data.cpu().numpy(), i_iter)

                # Discriminator 2
                for tag, value in model_D2.named_parameters():
                    tag = tag.replace('.', '/')
                    if value.grad is None:
                        # print('Dis2_Layer: ', tag.split('/'))
                        writer.add_histogram('dis_weights2/' + tag, value.data.cpu().numpy(), i_iter)
                        pass
                    else:
                        writer.add_histogram('dis_weights2/' + tag, value.data.cpu().numpy(), i_iter)
                        writer.add_histogram('dis_grads2/' + tag, value.grad.data.cpu().numpy(), i_iter)

    if i_iter >= config.NUM_STEPS - 1:
        print("Saving final model, mIoU={:.5} ..".format(mIoU))
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH + "final_saved_model_{:.5}_mIoU.pth".format(mIoU))

if __name__ == '__main__':
    main()
