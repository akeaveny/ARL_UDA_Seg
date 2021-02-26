import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image
import cv2

import matplotlib.pyplot as plt

########################
########################
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/umd_affordance/tools/*/'
new_data_path = '/data/Akeaveny/Datasets/domain_adaptation/UMD/DR_2/'

objects = [
    'bowl_01/',
]

scenes = [
    # 'bench/',
    # 'floor/',
    # 'turn_table/',
    'dr/',
]

splits = [
    'train/',
    'val/',
]

cameras = [
    'Kinect/',
    'Xtion/',
    'ZED/',
]

image_exts = [
    # '.png',
    '.depth.png',
    # '.depth.16.png',
    # '.cs.png',
]

########################
########################

offset = 0
for scene in scenes:
    print(f'\n*** offset:{offset} ***')
    for image_ext in image_exts:
        # file_path = data_path + object + scene + split + camera + '??????' + image_ext
        file_path = data_path   + '*/'   + scene + '*/'  + '*/'   + '??????' + image_ext
        files = np.array(sorted(glob.glob(file_path)))
        print("\nLoaded files: ", len(files))
        print("File path: ", file_path)

        for idx, file in enumerate(files):
            random_int = np.random.randint(low=0, high=len(files))
            file = files[random_int]
            print('file: ', file)

            file_path = file.split(image_ext)[0]
            rgb = file_path + '.png'
            depth = file_path + '.depth.png'
            obj_label = file_path + '.cs.png'
            aff_label = file_path + '.cs.png'

            rgb = cv2.imread(rgb, -1)
            depth = cv2.imread(depth, -1)
            obj_label = cv2.imread(obj_label, -1)
            aff_label = cv2.imread(aff_label, -1)

            rgb = np.array(rgb, dtype=np.uint8)
            depth = np.array(depth, dtype=np.uint8)
            obj_label = np.array(obj_label, dtype=np.uint8)
            aff_label = np.array(aff_label, dtype=np.uint8)

            cv2.imshow('rgb', rgb)
            cv2.imshow('depth', depth)
            cv2.imshow('heatmap', cv2.applyColorMap(depth, cv2.COLORMAP_JET))
            cv2.imshow('obj_label', obj_label*40)
            cv2.imshow('aff_label', aff_label*40)
            cv2.waitKey(0)