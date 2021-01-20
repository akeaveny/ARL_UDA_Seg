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
from model.pretrained_deeplab_multi import DeepLabv3_plus_multi
from model.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from torch.utils.data import DataLoader, random_split, Subset

from torch.utils.tensorboard import SummaryWriter

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

### WORKSTATION (LOCALLY)
# DIR_PATH = '/home/akeaveny/catkin_ws/src/AdaptSegNet/'
# SNAPSHOTS_PATH = '/data/Akeaveny/weights/ARLGAN/AdaptSegNet'
### REMOTE (SSH)
DIR_PATH = '/home/akeaveny/AdaptSegNet/'
SNAPSHOTS_PATH = '.'

EXP_NAME = 'AdaptSegNet_GTA_SEGMENTATION_pretrained_deeplab_multi_1024x512'

START_FROM = 0
RESTORE_FROM = DIR_PATH + 'model/DeepLab_resnet_pretrained_init-f81d91e8.pth'

MODEL = 'DeepLab' # 'DeepLab' or 'pretrained_deeplab_multi'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = './data/Cityscapes/'
DATA_LIST_PATH = DIR_PATH + 'dataset/cityscapes_list/rgb_train_list.txt'
DATA_LABEL_PATH = DIR_PATH + 'dataset/cityscapes_list/labels_train_list.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = './data/GTA5/'
DATA_LIST_PATH_TARGET = DIR_PATH + 'dataset/gta5_list/rgb_train_list.txt'
DATA_LABELS_PATH_TARGET = DIR_PATH + 'dataset/gta5_list/labels_train_list.txt'
INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = SNAPSHOTS_PATH + '/snapshots/' + EXP_NAME + '/'
WEIGHT_DECAY = 0.0005

TENSORBOARD_UPDATE = 5

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()

args = get_arguments()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)

    return criterion(pred, label)

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def color_map():
    color_map_dic = {
    0:  [0, 0, 0],
    1:  [128, 128,   0],
    2:  [  0, 128, 128],
    3:  [128,   0, 128],
    4:  [128,   0,   0],
    5:  [  0, 128,   0],
    6:  [  0,   0, 128],
    7:  [255, 255,   0],
    8:  [255,   0, 255],
    9:  [  0, 255, 255],
    10: [255,   0,   0],
    11: [  0, 255,   0],
    12: [  0,   0, 255],
    13: [ 92,  112, 92],
    14: [  0,   0,  70],
    15: [  0,  60, 100],
    16: [  0,  80, 100],
    17: [  0,   0, 230],
    18: [119,  11,  32],
    19: [  0,   0, 121],
    20: [44, 77, 62],
    21: [128, 255, 128],
    22: [256, 128, 128],
    23: [128, 0, 255],
    24: [128, 255, 0],
    25: [0, 255, 0],
    26: [0, 0, 255],
    27: [255, 128, 0],
    28: [128, 0, 255],
    29: [128, 255, 255],
    30: [128, 255, 0],
    31: [0, 255, 128],
    32: [128, 128, 255],
    }
    return color_map_dic

def colorize_mask(instance_mask):

    ########################
    #  add color to masks
    ########################
    instance_to_color = color_map()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def main():
    """Create the model and start the training."""
    args = get_arguments()

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    if args.model == 'DeepLab':
        model = DeeplabMulti(num_classes=args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)

        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            # Scale.layer5.conv2d_list.3.weight
            i_parts = i.split('.')
            # print i_parts
            if not args.num_classes == 19 or not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                # print i_parts
        model.load_state_dict(new_params)

    model.train()
    model.cuda(args.gpu)
    cudnn.benchmark = True

    # if not os.path.exists(args.snapshot_dir):
    #     os.makedirs(args.snapshot_dir)

    ### TENSORBOARD
    writer = SummaryWriter(f'{args.snapshot_dir}')

    ######################
    # LOAD SYN DATASET
    ######################
    dataset = GTA5DataSet(rgb_list=DATA_LIST_PATH_TARGET, labels_list=DATA_LABELS_PATH_TARGET,
                    max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size,
                    scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)
    print("GTA5 Dataset has: {}".format(len(dataset)))
    assert (len(dataset) == NUM_STEPS)

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    train_loader_iter = enumerate(train_loader)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    for i_iter in range(START_FROM, args.num_steps):

        loss_seg_value1 = 0
        loss_seg_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        for sub_i in range(args.iter_size):

            _, batch = train_loader_iter.__next__()
            images, labels = batch
            images = Variable(images).cuda(args.gpu)

            pred1, pred2 = model(images)
            pred1 = interp(pred1)
            pred2 = interp(pred2)

            if i_iter != 0 and i_iter % TENSORBOARD_UPDATE == 0:
                ### TENSORBOARD
                ### gt
                gt = labels.cpu().detach().squeeze()
                gt = np.array(gt, dtype=np.int8)
                # gt = gt[np.newaxis, np.newaxis, :, :]
                gt = colorize_mask(gt)
                gt = np.transpose(gt, (2, 0, 1))
                gt = gt[np.newaxis, :, :, :]
                writer.add_images('source/gt_mask', gt, i_iter)
                ### pred
                pred = pred2.clone().cpu().detach().squeeze()
                pred = np.array(pred, dtype=np.int8)
                pred = np.transpose(pred, (1, 2, 0))
                pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
                # pred = pred[np.newaxis, np.newaxis, :, :]
                pred = colorize_mask(pred)
                pred = np.transpose(pred, (2, 0, 1))
                pred = pred[np.newaxis, :, :, :]
                writer.add_images('source/pred_mask', pred, i_iter)

            loss_seg1 = loss_calc(pred1, labels, args.gpu)
            loss_seg2 = loss_calc(pred2, labels, args.gpu)
            loss = loss_seg2 + args.lambda_seg * loss_seg1

            # proper normalization
            loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_seg1.data.cpu().numpy() / args.iter_size
            loss_seg_value2 += loss_seg2.data.cpu().numpy() / args.iter_size

            ### TENSORBOARD
            writer.add_scalar('Loss/train', loss_seg_value1 + loss_seg_value2, i_iter)
            writer.add_scalar('Loss/seg_loss1', loss_seg_value1, i_iter)
            writer.add_scalar('Loss/seg_loss2', loss_seg_value2, i_iter)

        optimizer.step()

        print('exp = {}'.format(args.snapshot_dir))
        print('iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f}'.format(i_iter, args.num_steps,
                                                                                     loss_seg_value1, loss_seg_value2))

        ## TENSORBOARD
        writer.add_scalar('learning_rate/seg', optimizer.param_groups[0]['lr'], i_iter)
        # hist and dist
        if i_iter != 0 and i_iter % TENSORBOARD_UPDATE == 0:
            # segmentation
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                if value.grad is None:
                    # print('Seg_Layer: ', tag.split('/'))
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), i_iter)
                    pass
                else:
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), i_iter)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), i_iter)

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            break

        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))

if __name__ == '__main__':
    main()
