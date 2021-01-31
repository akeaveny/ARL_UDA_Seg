import numpy as np

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1].absolute().resolve(strict=True)

import config

######################
######################

def print_class_labels(seg_name, seg_mask):
    class_ids = np.unique(np.array(seg_mask, dtype=np.uint8))
    print(f"{seg_name} has {len(class_ids)} Labels:\n{class_ids}")

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

def numpy_2_torch(numpy_img, mean=config.IMG_MEAN, is_rgb=False):
    torch_img = np.asarray(numpy_img, np.float32)

    if is_rgb:
        torch_img = torch_img[:, :, ::-1]               # change to BGR
        torch_img -= np.array(mean, dtype=np.float32)
        torch_img = torch_img.transpose((2, 0, 1))      # images are represented as [C, H W] in torch

    return torch_img

def torch_2_numpy(torch_img, mean=config.IMG_MEAN, is_rgb=False):
    numpy_img = np.squeeze(np.array(torch_img, dtype=np.uint8))

    if is_rgb:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))  # images are represented as [C, H W] in torch
        numpy_img += np.array(mean, dtype=np.uint8)
        numpy_img = numpy_img[:, :, ::-1]               # change to BGR

    return numpy_img

def cuda_2_numpy(cuda_img,  mean=config.IMG_MEAN, is_rgb=False, is_pred=False):
    numpy_img = cuda_img.cpu().detach().squeeze()
    numpy_img = np.array(numpy_img)

    if is_rgb:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))  # images are represented as [C, H W] in torch
        numpy_img += np.array(mean, dtype=np.uint8)
        numpy_img = numpy_img[:, :, ::-1]  # change to BGR

    if is_pred:
        numpy_img = np.transpose(numpy_img, (1, 2, 0))
        numpy_img = np.asarray(np.argmax(numpy_img, axis=2), dtype=np.uint8)

    return np.array(numpy_img, dtype=np.uint8)


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

    instance_to_color = color_map()
    color_mask = np.zeros((instance_mask.shape[0], instance_mask.shape[1], 3), dtype=np.uint8)
    for key in instance_to_color.keys():
        color_mask[instance_mask == key] = instance_to_color[key]

    return np.squeeze(color_mask)