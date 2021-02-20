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

def train_AdaptSegNet_multi(model, model_D1, model_D2,target_loader, source_loader,test_loader,writer):
    """TRAIN WITH DOMAIN ADAPTATION USING ADAPTSEGNET"""

    ######################
    # DIS
    ######################

    upsample_source = nn.Upsample(size=config.INPUT_SIZE, mode='bilinear', align_corners=True)
    upsample_target = nn.Upsample(size=config.INPUT_SIZE_TARGET, mode='bilinear', align_corners=True)

    ######################
    # LOSS
    ######################

    criterion = CrossEntropy2d().cuda(config.GPU)

    if config.GAN == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif config.GAN == 'LS':
        bce_loss = torch.nn.MSELoss()

    ######################
    # OPTIMIZER
    ######################

    optimizer = optim.SGD(model.optim_parameters(config.LEARNING_RATE),
                          lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=config.LEARNING_RATE_D, betas=(config.BETA_1, config.BETA_2))
    optimizer_D2 = optim.Adam(model_D2.parameters(), lr=config.LEARNING_RATE_D, betas=(config.BETA_1, config.BETA_2))

    optimizer.zero_grad()
    optimizer_D1.zero_grad()
    optimizer_D2.zero_grad()

    ######################
    # EVAL METRIC
    ######################

    from utils.eval_metrics import eval_model
    # mIoU, best_mIoU = -np.inf, -np.inf
    Fwb, best_Fwb = -np.inf, -np.inf

    ######################
    ######################

    # labels for adversarial training
    source_label = 0
    target_label = 1

    i_iter = config.StartIteration
    NUM_STEPS = config.NUM_STEPS_STOP if config.NUM_STEPS_STOP < config.NUM_STEPS else config.NUM_STEPS
    with tqdm(total=NUM_STEPS - config.StartIteration, desc=f'Iterations {NUM_STEPS}',
              unit='images') as pbar:
        while i_iter < NUM_STEPS:

            loss_seg_value1 = 0
            loss_adv_target_value1 = 0
            loss_D_value1 = 0

            loss_seg_value2 = 0
            loss_adv_target_value2 = 0
            loss_D_value2 = 0

            optimizer.zero_grad()
            helper_utils.adjust_learning_rate(optimizer, i_iter)

            optimizer_D1.zero_grad()
            optimizer_D2.zero_grad()
            helper_utils.adjust_learning_rate_D(optimizer_D1, i_iter)
            helper_utils.adjust_learning_rate_D(optimizer_D2, i_iter)

            ######################
            # train G
            ######################

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False

            for param in model_D2.parameters():
                param.requires_grad = False

            # train with source
            _, batch = source_loader.__next__()
            images, labels, depths = batch['image'], batch['label'], batch['depth']
            images = images.to(device=config.GPU, dtype=torch.float32)
            labels = labels.to(device=config.GPU, dtype=torch.long)
            depths = depths.to(device=config.GPU, dtype=torch.float32)
            source_img, source_depth, source_gt = images[0, :, :, :], depths[0, :, :, :], labels[0, :, :]
            # for depth based training
            if config.NUM_CHANNELS == 1:
                images = depths

            if i_iter == 0:
                if config.MODEL == 'DeepLabMulti' or config.MODEL == 'DeepLabv3Multi':
                    writer.add_graph(model, images)
                elif config.MODEL == 'DeepLabv3DepthMulti':
                    writer.add_graph(model, [images, depths])

            if config.MODEL == 'DeepLabMulti' or config.MODEL == 'DeepLabv3Multi':
                pred_source_aux, pred_source_main = model(images)
            elif config.MODEL == 'DeepLabv3DepthMulti':
                pred_source_aux, pred_source_main = model(images, depths)
            source_pred = pred_source_main[0, :, :]

            loss_seg1 = criterion(pred_source_aux, labels)
            loss_seg2 = criterion(pred_source_main, labels)
            loss_seg = loss_seg2 + config.LAMBDA_SEG * loss_seg1

            # proper normalization
            loss = loss_seg
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy()
            loss_seg_value2 += loss_seg2.data.cpu().numpy()

            ### TENSORBOARD
            writer.add_scalar('SegLoss/SegLoss', loss_seg_value1 + loss_seg_value2, i_iter)
            writer.add_scalar('SegLoss/loss_seg_value1', loss_seg_value1, i_iter)
            writer.add_scalar('SegLoss/loss_seg_value2', loss_seg_value2, i_iter)

            ######################
            ######################

            # train with target
            _, batch = target_loader.__next__()
            images, labels, depths = batch['image'], batch['label'], batch['depth']
            images = images.to(device=config.GPU, dtype=torch.float32)
            labels = labels.to(device=config.GPU, dtype=torch.long)
            depths = depths.to(device=config.GPU, dtype=torch.float32)
            target_img, target_depth, target_gt = images[0, :, :, :], depths[0, :, :, :], labels[0, :, :]
            # for depth based training
            if config.NUM_CHANNELS == 1:
                images = depths

            if config.MODEL == 'DeepLabMulti' or config.MODEL == 'DeepLabv3Multi':
                pred_target_aux, pred_target_main = model(images)
            elif config.MODEL == 'DeepLabv3DepthMulti':
                pred_target_aux, pred_target_main = model(images, depths)
            target_pred = pred_target_main[0, :, :]

            D_out1 = model_D1(F.softmax(pred_target_aux))
            D_out2 = model_D2(F.softmax(pred_target_main))

            loss_adv_target1 = bce_loss(D_out1, helper_utils.fill_DA_label(D_out1.data.size(), source_label))
            loss_adv_target2 = bce_loss(D_out2, helper_utils.fill_DA_label(D_out2.data.size(), source_label))
            loss_adv = config.LAMBDA_ADV_TARGET1 * loss_adv_target1 + config.LAMBDA_ADV_TARGET2 * loss_adv_target2

            loss = loss_adv
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.data.cpu().numpy()
            loss_adv_target_value2 += loss_adv_target2.data.cpu().numpy()

            ### TENSORBOARD
            writer.add_scalar('ADVLoss/ADVLoss', loss_adv_target_value1 + loss_adv_target_value2, i_iter)
            writer.add_scalar('ADVLoss/seg_adv_loss1', loss_adv_target_value1, i_iter)
            writer.add_scalar('ADVLoss/seg_adv_loss2', loss_adv_target_value2, i_iter)

            ######################
            # train D
            ######################

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred_source_aux = pred_source_aux.detach()
            pred_source_main = pred_source_main.detach()

            D_out1 = model_D1(F.softmax(pred_source_aux))
            D_out2 = model_D2(F.softmax(pred_source_main))

            loss_D1 = bce_loss(D_out1, helper_utils.fill_DA_label(D_out1.data.size(), source_label))
            loss_D2 = bce_loss(D_out2, helper_utils.fill_DA_label(D_out2.data.size(), source_label))

            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2

            loss_D1.backward()
            loss_D2.backward()

            loss_D_value1 += loss_D1.data.cpu().numpy()
            loss_D_value2 += loss_D2.data.cpu().numpy()

            # train with target
            pred_target_aux = pred_target_aux.detach()
            pred_target_main = pred_target_main.detach()

            D_out1 = model_D1(F.softmax(pred_target_aux))
            D_out2 = model_D2(F.softmax(pred_target_main))

            loss_D1 = bce_loss(D_out1, helper_utils.fill_DA_label(D_out1.data.size(), target_label))
            loss_D2 = bce_loss(D_out2, helper_utils.fill_DA_label(D_out2.data.size(), target_label))

            loss_D1 = loss_D1 / 2
            loss_D2 = loss_D2 / 2

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
                    torch.save(model_D1.state_dict(), config.BEST_DIS1_SAVE_PATH)
                    torch.save(model_D2.state_dict(), config.BEST_DIS2_SAVE_PATH)
                    ### TENSORBOARD: loading best images
                    writer.add_images('source_img/rgb',
                                      helper_utils.cuda_img_2_tensorboard(source_img),
                                      i_iter)
                    writer.add_images('source_img/depth',
                                      helper_utils.cuda_img_2_tensorboard(source_depth, is_depth=True),
                                      i_iter)
                    writer.add_images('source/gt_mask',
                                      helper_utils.cuda_label_2_tensorboard(source_gt),
                                      i_iter)
                    writer.add_images('source/pred_mask',
                                      helper_utils.cuda_label_2_tensorboard(source_pred, is_pred=True),
                                      i_iter)
                    writer.add_images('target_img/rgb',
                                      helper_utils.cuda_img_2_tensorboard(target_img),
                                      i_iter)
                    writer.add_images('target_img/depth',
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
            ## TENSORBOARD
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

            if i_iter != 0 and i_iter % config.SAVE_PRED_EVERY == 0:
                print(f'Saved Model at {i_iter}..')
                torch.save(model.state_dict(),    config.MODEL_SAVE_PATH + config.EXP_DATASET_NAME + str(i_iter) + '.pth')
                c

    if i_iter >= NUM_STEPS - 1:
        print("Saving final model, mIoU={:.5} ..".format(mIoU))
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH + "final_saved_model_{:.5}_mIoU.pth".format(mIoU))