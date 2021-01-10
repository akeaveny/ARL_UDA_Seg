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
data_path = '/data/Akeaveny/Datasets/domain_adaptation/GTA5/'
new_data_path = '/data/Akeaveny/Datasets/domain_adaptation/ARLGAN/GTA5/'

splits = ['']

image_exts = [
    '.png',
]

########################
########################
for split in splits:
    offset = 0
    for image_ext in image_exts:
        file_path = data_path + split + '*/' 'images/' + '*' + image_ext
        # file_path = data_path + split + '*/' 'labels/' + '*' + image_ext
        print("File path: ", file_path)
        files = np.array(sorted(glob.glob(file_path)))
        print("Loaded files: ", len(files))

        ###################
        ###################

        for idx, file in enumerate(files):
            old_file_name = file
            folder_to_move = new_data_path + split

            count = 1000000 + offset + idx
            image_num = str(count)[1:]
            print(f'\nImage num {image_num}')

            if image_ext == '.png':
                move_file_name = folder_to_move + 'rgb/' + np.str(image_num) + '.png'
                print(f'Old file: {old_file_name}')
                print(f'New file: {move_file_name}')
                shutil.copyfile(old_file_name, move_file_name)

            # if image_ext == '.png':
            #     move_file_name = folder_to_move + 'masks/' + np.str(image_num) + '_label.png'
            #     print(f'Old file: {old_file_name}')
            #     print(f'New file: {move_file_name}')
            #     shutil.copyfile(old_file_name, move_file_name)
            #
            #     # gt_label = np.array(Image.open(old_file_name))
            #     # print("gt_label:", len(np.unique(gt_label)[1:]))
            #     # print("gt_label:", np.unique(gt_label)[1:])
            #
            #     # plt.title("gt_label")
            #     # plt.imshow(gt_label)
            #     # plt.show()
            #     # plt.ioff()

            else:
                print("*** IMAGE EXT DOESN'T EXIST ***")
                exit(1)