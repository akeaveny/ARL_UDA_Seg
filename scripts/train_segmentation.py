import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

###############################
###############################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1].absolute().resolve(strict=True)

import cfg as config

from utils import helper_utils
from utils.loss import CrossEntropy2d

from tqdm import tqdm

###############################
###############################

def train_segmentation(model, target_loader, test_loader, writer):
    """TRAIN SEG NETWORK WITHOUT DOMAIN ADAPTATION"""

    ######################
    # LOSS
    ######################

    criterion = CrossEntropy2d().cuda(config.GPU)

    ######################
    # OPTIMIZER
    ######################

    optimizer = optim.SGD(model.optim_parameters(config.LEARNING_RATE),
                          lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    optimizer.zero_grad()

    ######################
    # EVAL METRIC
    ######################

    from utils.eval_metrics import eval_model
    # mIoU, best_mIoU = -np.inf, -np.inf
    Fwb = -np.inf
    best_Fwb = config.BestFwb if config.BestFwb is not None else -np.inf

    ######################
    ######################
    i_iter = config.StartIteration
    with tqdm(total=config.NUM_STEPS-config.StartIteration, desc=f'Iterations {config.NUM_STEPS}', unit='images') as pbar:
        while i_iter < config.NUM_STEPS:

            loss_seg_value = 0

            optimizer.zero_grad()
            helper_utils.adjust_learning_rate(optimizer, i_iter)

            _, batch = target_loader.__next__()
            images, labels, depths = batch['image'], batch['label'], batch['depth']
            images = images.to(device=config.GPU, dtype=torch.float32)
            labels = labels.to(device=config.GPU, dtype=torch.long)
            depths = depths.to(device=config.GPU, dtype=torch.float32)
            target_img, target_depth, target_gt = images[0, :, :, :], depths[0, :, :, :], labels[0, :, :]
            # for depth based training
            if config.NUM_CHANNELS == 1:
                images = depths

            if i_iter == 0:
                if config.MODEL == 'DeepLab' or config.MODEL == 'DeepLabv3':
                    writer.add_graph(model, images)
                elif config.MODEL == 'DeepLabDepth' or config.MODEL == 'DeepLabv3Depth':
                    writer.add_graph(model, [images, depths])

            if config.MODEL == 'DeepLab' or config.MODEL == 'DeepLabv3':
                pred_target_main = model(images)
            elif config.MODEL == 'DeepLabDepth' or config.MODEL == 'DeepLabv3Depth':
                pred_target_main = model(images, depths)
            target_pred = pred_target_main[0, :, :]

            loss_seg = criterion(pred_target_main, labels)

            loss = loss_seg
            loss.backward()
            loss_seg_value += loss_seg.data.cpu().numpy()

            ### TENSORBOARD
            writer.add_scalar('SegLoss/SegLoss', loss_seg_value, i_iter)

            optimizer.step()

            pbar.set_postfix(**{'SegLoss: ': loss_seg_value})

            i_iter += 1
            pbar.update(int(images.shape[0]/config.BATCH_SIZE))

            ## EVAL
            if i_iter != 0 and i_iter % config.EVAL_UPDATE == 0:
                # mIoU = eval_model(model, test_loader, eval_mIoU=True)
                # writer.add_scalar('eval/mIoU', mIoU, i_iter)
                Fwb = eval_model(model, test_loader, eval_Fwb=True)
                writer.add_scalar('eval/Fwb', Fwb, i_iter)
                if Fwb > best_Fwb:
                    best_Fwb = Fwb
                    writer.add_scalar('eval/Best Fwb', best_Fwb, i_iter)
                    print("Saving best model .. best Fwb={:.5} ..".format(best_Fwb))
                    torch.save(model.state_dict(), config.BEST_MODEL_SAVE_PATH)
                    ### TENSORBOARD: loading best images
                    writer.add_images('dataset/target_img',
                                      helper_utils.cuda_img_2_tensorboard(target_img),
                                      i_iter)
                    writer.add_images('dataset/target_depth',
                                      helper_utils.cuda_img_2_tensorboard(target_depth, is_depth=True),
                                      i_iter)
                    writer.add_images('target/gt_mask',
                                      helper_utils.cuda_label_2_tensorboard(target_gt),
                                      i_iter)
                    writer.add_images('target/pred_mask',
                                      helper_utils.cuda_label_2_tensorboard(target_pred, is_pred=True),
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

            if i_iter != 0 and i_iter % config.SAVE_PRED_EVERY == 0:
                print(f'Saved Model at {i_iter}..')
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH + config.EXP_DATASET_NAME + str(i_iter) + '.pth')

    if i_iter >= config.NUM_STEPS - 1:
        # print("Saving final model, mIoU={:.5} ..".format(mIoU))
        # torch.save(model.state_dict(), config.MODEL_SAVE_PATH + "final_saved_model_{:.5}_mIoU.pth".format(mIoU))
        print("Saving final model, Fwb={:.5} ..".format(Fwb))
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH + "final_saved_model_{:.5}_Fwb.pth".format(Fwb))