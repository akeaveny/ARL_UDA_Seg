import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import transforms

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1].absolute().resolve(strict=True)

import cfg as config

######################
######################

def print_depth_info(depth):
    depth = np.array(depth)
    print(f"Depth of type:{depth.dtype} has min:{np.min(depth)} & max:{np.max(depth)}")

######################
######################

def print_class_labels(seg_mask):
    class_ids = np.unique(np.array(seg_mask, dtype=np.uint8))
    print(f"Mask has {len(class_ids)-1} Labels: {class_ids[1:]}")

gta5_id_to_cityscapes_id = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                            19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                            26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}

def gta5_to_cityscapes_label(gta5_label):
    gta5_label = np.array(gta5_label, dtype=np.uint8)
    cityscapes_label = 255 * np.ones(shape=(gta5_label.shape), dtype=np.float32)

    for k, v in gta5_id_to_cityscapes_id.items():
        cityscapes_label[gta5_label == k] = v

    return np.array(cityscapes_label, dtype=np.uint8)

######################
######################

def numpy_2_torch(numpy_img, mean=config.IMG_MEAN, is_rgb=False, is_depth=False):
    torch_img = np.asarray(numpy_img, np.float32)

    if is_rgb:
        torch_img = torch_img[:, :, ::-1]               # change to BGR
        torch_img -= np.array(mean[0:-1], dtype=np.float32)
        torch_img = torch_img.transpose((2, 0, 1))      # images are represented as [C, H, W] in torch

    if is_depth:
        torch_img = torch_img[np.newaxis, :, :]
        mean_ = np.array(mean[-1], dtype=np.float32)
        torch_img -= mean_

    return torch_img

def torch_2_numpy(torch_img, mean=config.IMG_MEAN, is_rgb=False, is_depth=False):

        numpy_img = np.squeeze(np.array(torch_img, dtype=np.uint8))

        if is_rgb:
            numpy_img = np.transpose(numpy_img, (1, 2, 0))  # images are represented as [C, H W] in torch
            numpy_img += np.array(mean[0:-1], dtype=np.uint8)
            numpy_img = numpy_img[:, :, ::-1]               # change to BGR

        if is_depth:
            mean_ = np.array(mean[-1], dtype=np.uint8)
            numpy_img += mean_

        return np.array(numpy_img, dtype=np.uint8)

def cuda_2_numpy(cuda_img, mean=config.IMG_MEAN, is_rgb=False, is_pred=False):
    numpy_img = cuda_img.squeeze().cpu().detach().numpy()

    if is_rgb:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))  # images are represented as [C, H W] in torch
        numpy_img += np.array(mean[0:-1], dtype=np.uint8)
        numpy_img = numpy_img[:, :, ::-1]  # change to BGR

    if is_pred:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))
        numpy_img = np.asarray(np.argmax(numpy_img, axis=2), dtype=np.uint8)

        # probs = F.softmax(cuda_img, dim=1)
        # numpy_img = probs.squeeze().cpu().detach().numpy()
        # numpy_img = [numpy_img[c, :, :] > config.CONFIDENCE_THRESHOLD for c in range(1, numpy_img.shape[0])]
        # numpy_img = np.asarray(np.argmax(np.asarray(numpy_img), axis=0)+1, dtype=np.uint8)

    return np.array(numpy_img, dtype=np.uint8)

def fill_DA_label(size, label):
    return Variable(torch.FloatTensor(size).fill_(label)).cuda(config.GPU)

######################
######################

def cuda_img_2_tensorboard(cuda_img, is_depth=False):

    img = cuda_img.cpu().detach().squeeze()
    # now format for to [BS, C, H W] in tensorboard
    if is_depth:
        return np.array(img)[np.newaxis, np.newaxis, :, :]
    else:
        return np.array(img)[np.newaxis, :, :, :]

def cuda_label_2_tensorboard(cuda_label, is_pred=False):

    colour_label = cuda_label.cpu().detach().squeeze()
    colour_label = np.array(colour_label)
    if is_pred:
        colour_label = np.transpose(colour_label, (1, 2, 0))
        colour_label = np.asarray(np.argmax(colour_label, axis=2), dtype=np.uint8)
    colour_label = np.array(colour_label, dtype=np.uint8)
    colour_label = colorize_mask(colour_label)
    # now format for to [BS, C, H W] in tensorboard
    return np.transpose(colour_label, (2, 0, 1))[np.newaxis, :, :, :]

######################
######################

def colorize_mask(instance_mask):

    instance_to_color = color_map()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)

def color_map():
    ''' [red, blue, green]'''

    color_map_dic = {
    0:  [0, 0, 0],
    1:  [235, 17,  17],     # grasp: red
    2:  [235, 209, 17],     # cut: yellow
    3:  [113, 235, 17],     # scoop: green
    4:  [17,  235, 202],    # contain: teal
    5:  [17,   54, 235],    # pound: blue
    6:  [129,  17, 235],    # support: purple
    7:  [235,  17, 179],    # wrap-grasp: pink
    # TODO: fill rest of colors
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

######################
######################

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(config.LEARNING_RATE, i_iter, config.NUM_STEPS, config.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(config.LEARNING_RATE_D, i_iter, config.NUM_STEPS, config.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

######################
######################

def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)

def adjust_learning_rate_CLAN(optimizer, i_iter):
    if i_iter < config.PREHEAT_STEPS:
        lr = lr_warmup(config.LEARNING_RATE, i_iter, config.PREHEAT_STEPS)
    else:
        lr = lr_poly(config.LEARNING_RATE, i_iter, config.NUM_STEPS, config.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def adjust_learning_rate_D_CLAN(optimizer, i_iter):
    if i_iter < config.PREHEAT_STEPS:
        lr = lr_warmup(config.LEARNING_RATE_D, i_iter, config.PREHEAT_STEPS)
    else:
        lr = lr_poly(config.LEARNING_RATE_D, i_iter, config.NUM_STEPS, config.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(config.BATCH_SIZE, 1, pred1.size(2), pred1.size(3)) / \
    (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(config.BATCH_SIZE, 1, pred1.size(2), pred1.size(3))
    return output