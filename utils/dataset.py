from os.path import splitext
from os import listdir
from glob import glob

import logging

import numpy as np

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import skimage.transform
from skimage.util import crop

import torch
from torch.utils import data
from torch.utils.data import Dataset
import torchvision

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1].absolute().resolve(strict=True)

import config
from utils import helper_utils

######################
######################

class BasicDataSet(data.Dataset):

    def __init__(self,
                 dataset_dir,
                 gta5_remap_label_idx=False,
                 use_depth_imgs=False,
                 ### FOLDER MUST BE CORRECTLY FORMATTED
                 rgb_dir='rgb/',
                 rgb_suffix='',
                 masks_dir='masks/',
                 masks_suffix='_label',
                 depth_dir='depth/',
                 depth_suffix='_depth',
                 ### EXTENDING DATASET
                 extend_dataset=False,
                 max_iters=int(250e3),
                 ### PRE-PROCESSING
                 mean=(128, 128, 128),
                 crop_size=(512, 256),
                 ignore_label=255,
                 ### IMGAUG
                 apply_imgaug=False):

        self.dataset_dir = dataset_dir
        self.gta5_remap_label_idx = gta5_remap_label_idx
        self.use_depth_imgs = use_depth_imgs
        ### FOLDER MUST BE CORRECTLY FORMATTED
        self.rgb_dir = self.dataset_dir + rgb_dir
        self.rgb_suffix = rgb_suffix
        self.masks_dir = self.dataset_dir + masks_dir
        self.masks_suffix = masks_suffix
        if self.use_depth_imgs:
            self.depth_dir = self.dataset_dir + depth_dir
            self.depth_suffix = depth_suffix
        ### pre-processing
        self.mean = mean
        self.crop_size = crop_size
        self.ignore_label = ignore_label

        ################################
        ### EXTENDING DATASET
        ################################
        self.extend_dataset = extend_dataset
        self.max_iters = max_iters

        self.rgb_ids = [splitext(file)[0] for file in listdir(self.rgb_dir) if not file.startswith('.')]
        self.masks_ids = [splitext(file)[0] for file in listdir(self.masks_dir) if not file.startswith('.')]
        assert(len(self.rgb_ids) == len(self.masks_ids))
        print(f'Original dataset has {len(self.rgb_ids)} examples')

        # creating larger dataset
        if self.extend_dataset:
            ids = []
            total_idx = np.arange(0, len(self.rgb_ids), 1)
            for image_idx in range(self.max_iters):
                idx = np.random.choice(total_idx, size=1, replace=False)
                ids.append(self.rgb_ids[int(idx)])
            self.rgb_ids = ids
            print(f'Extended dataset has {len(self.rgb_ids)} examples')

        ################################
        # IMGAUG
        ################################
        self.apply_imgaug = apply_imgaug

        self.affine = iaa.Sequential([
            iaa.Fliplr(0.5),   # horizontally flip 50% of the images
            # iaa.Flipud(0.5), # vertical flip 50% of the images
            # iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
        ], random_order=True)

        self.colour_aug = iaa.Sometimes(0.833, iaa.Sequential([
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            # Strengthen or weaken the contrast in each image.
            iaa.contrast.LinearContrast((0.75, 1.25)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
        ], random_order=True))  # apply augmenters in random order

        self.depth_aug = iaa.Sometimes(0.833, iaa.Sequential([
            iaa.CoarseDropout(0.02, size_percent=0.5),
            iaa.SaltAndPepper(0.1),
        ], random_order=True))  # apply augmenters in random order

    def __len__(self):
        return len(self.rgb_ids)

    def apply_imgaug_to_imgs(self, rgb, mask, depth=None):
        rgb, depth, mask = np.array(rgb), np.array(depth), np.array(mask)

        H, W, C = rgb.shape[0], rgb.shape[1], rgb.shape[2]

        concat_img = np.zeros(shape=(H, W, C + 1))
        concat_img[:, :, :C] = rgb
        concat_img[:, :, -1] = depth[:, :]
        concat_img = np.array(concat_img, dtype=np.uint8)

        segmap = SegmentationMapsOnImage(mask, shape=np.array(rgb).shape)
        aug_concat_img, segmap = self.affine(image=concat_img, segmentation_maps=segmap)
        mask = segmap.get_arr()

        rgb = aug_concat_img[:, :, :C]
        depth = aug_concat_img[:, :, -1]
        depth = np.array(skimage.color.gray2rgb(depth))

        rgb = self.colour_aug(image=rgb)
        depth = self.depth_aug(image=depth)

        rgb, mask = np.array(rgb), np.array(mask)
        depth = np.array(depth) if self.use_depth_imgs else np.zeros(shape=(rgb.shape[0], rgb.shape[1]))

        return rgb, mask, depth

    def __getitem__(self, index):

        idx = self.rgb_ids[index]
        img_file = glob(self.rgb_dir + idx + self.rgb_suffix + '.*')
        mask_file = glob(self.masks_dir + idx + self.masks_suffix + '.*')

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {idx}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        image = Image.open(img_file[0]).convert('RGB')
        image = image.resize(self.crop_size, Image.BICUBIC)

        label = Image.open(mask_file[0])
        label = label.resize(self.crop_size, Image.NEAREST)

        ##################
        ### TODO: DEPTH
        ##################
        if self.use_depth_imgs:
            depth_file = glob(self.depth_dir + idx + self.depth_suffix + '.*')
            assert len(depth_file) == 1, f'Either no image or multiple images found for the ID {idx}: {depth_file}'
            depth = Image.open(depth_file[0])
        else:
            depth = Image.fromarray(np.zeros(shape=(image.size[1], image.size[0])))

        ##################
        ### IMGAUG
        ##################

        if self.apply_imgaug:
            image, label, depth = self.apply_imgaug_to_imgs(rgb=image, mask=label, depth=depth)

        ##################
        ### REMAP GTA
        ##################

        if self.gta5_remap_label_idx:
            label = helper_utils.gta5_to_cityscapes_label(label)

        ##################
        ### SEND TO TORCH
        ##################

        # print("Num Classes: {}".format(np.unique(label)))

        image = helper_utils.numpy_2_torch(image, mean=self.mean, is_rgb=True)
        label = helper_utils.numpy_2_torch(label)
        depth = helper_utils.numpy_2_torch(skimage.color.gray2rgb(np.array(depth)))     # triplcate depth to match RGB

        return {
            'image': image.copy(),
            'label': label.copy(),
            'depth': depth.copy()
        }

if __name__ == '__main__':
    dst = BasicDataSet(dataset_dir=config.DATA_DIRECTORY_SOURCE_TRAIN,
                       gta5_remap_label_idx=True,
                       extend_dataset=False,
                       apply_imgaug=False)
    trainloader = data.DataLoader(dst, batch_size=1)
    print("Cityscapes Dataset has: {}".format(len(trainloader)))
    for i, data in enumerate(trainloader):
        imgs, labels, depths = data['image'], data['label'], data['depth']
        ###
        imgs = helper_utils.torch_2_numpy(imgs, mean=dst.mean, is_rgb=True)
        labels = helper_utils.torch_2_numpy(labels)
        depths = helper_utils.torch_2_numpy(depths)
        ### cv2
        cv2.imshow('rgb', cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
        cv2.imshow('label', labels)
        if dst.use_depth_imgs:
            cv2.imshow('depth', depths)
        cv2.waitKey(0)