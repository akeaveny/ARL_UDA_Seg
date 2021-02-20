import numpy as np
import shutil
import glob
import os

import scipy.io
import scipy.misc
from PIL import Image

import matplotlib.pyplot as plt

########################
########################
data_path = '/data/Akeaveny/Datasets/part-affordance_combined/ndds4/combined_tools5_'
new_data_path = '/data/Akeaveny/Datasets/domain_adaptation/UMD/Syn/'

splits = [
    'train/',
    'val/',
    'test/',
]

scenes = [
    'bench/',
    'floor/',
    'turn_table/',
    'dr/',
]

image_exts = [
            '_rgb.png',
            '_depth.png',
            '_gt_affordance.png'
]

########################
########################
offset = 0
for split in splits:
    for scene in scenes:
        print(f'\n*** offset:{offset} ***')
        for image_ext in image_exts:
            file_path = data_path + split + scene + '*' + image_ext
            files = np.array(sorted(glob.glob(file_path)))
            print("\nLoaded files: ", len(files))
            print("File path: ", file_path)

            ###################
            ###################

            for idx, file in enumerate(files):
                old_file_name = file
                new_file_name = new_data_path + 'train/'

                count = 1000000 + offset + idx
                image_num = str(count)[1:]

                if image_ext == '_rgb.png':
                    move_file_name = new_file_name + 'rgb/' + np.str(image_num) + '.png'
                    if idx == 0 and split == 'train/':
                        print(f'Old file: {old_file_name}')
                        print(f'New file: {move_file_name}')
                    shutil.copyfile(old_file_name, move_file_name)

                elif image_ext == '_depth.png':
                    move_file_name = new_file_name + 'depth/' + np.str(image_num) + '_depth.png'
                    if idx == 0 and split == 'train/':
                        print(f'Old file: {old_file_name}')
                        print(f'New file: {move_file_name}')
                    shutil.copyfile(old_file_name, move_file_name)

                elif image_ext == '_gt_affordance.png':
                    move_file_name = new_file_name + 'masks/' + np.str(image_num) + '_label.png'
                    if idx == 0 and split == 'train/':
                        print(f'Old file: {old_file_name}')
                        print(f'New file: {move_file_name}')
                    shutil.copyfile(old_file_name, move_file_name)

                else:
                    print("*** IMAGE EXT DOESN'T EXIST ***")
                    exit(1)

            ###################
            ###################

            if image_ext == '_gt_affordance.png':
                offset += len(files)