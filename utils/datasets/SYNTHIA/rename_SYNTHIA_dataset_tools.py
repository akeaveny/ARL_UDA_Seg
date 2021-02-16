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
data_path = '/data/Akeaveny/Datasets/domain_adaptation/SYNTHIA/RAND_CITYSCAPES/'
new_data_path = '/data/Akeaveny/Datasets/domain_adaptation/ARLGAN/SYNTHIA/'

splits = [
        'RGB/',
        'Depth/',
        'GT/LABELS/',
        ]

image_exts = [
    '.png',
]

########################
########################
for split in splits:
    offset = 0
    for image_ext in image_exts:
        file_path = data_path + split + '*' + image_ext
        # print("File path: ", file_path)
        files = np.array(sorted(glob.glob(file_path)))
        print("Loaded files: ", len(files))

        ###################
        ###################

        for idx, file in enumerate(files):
            old_file_name = file
            folder_to_move = new_data_path

            count = 1000000 + offset + idx
            image_num = str(count)[1:]
            # print(f'\nImage num {image_num}')

            if split == 'RGB/' and image_ext == '.png':
                move_file_name = folder_to_move + 'rgb/' + np.str(image_num) + '.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif split == 'Depth/' and image_ext == '.png':
                move_file_name = folder_to_move + 'depth/' + np.str(image_num) + '_depth.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            elif split == 'GT/LABELS/' and image_ext == '.png':
                move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_label.png'
                if idx == 0:
                    print(f'Old file: {old_file_name}')
                    print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            else:
                print("*** IMAGE EXT DOESN'T EXIST ***")
                exit(1)