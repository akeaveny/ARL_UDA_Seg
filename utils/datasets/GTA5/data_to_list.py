import numpy as np
import shutil
import glob
import os

from PIL import Image
import matplotlib.pyplot as plt

# =================== new directory ========================
# 0.
data_path = '/home/akeaveny/datasets/ARLGAN/GTA5/'
list_path = '/home/akeaveny/git/ARL_UDA_Seg/dataset/gta5_list/'

rgb_train_file = open(list_path + 'rgb_train_list.txt', 'w')
labels_train_file = open(list_path + 'labels_train_list.txt', 'w')

splits = ['']

image_exts1 = ['']

# creating larger dataset
np.random.seed(0)
extend_dataset = True
num_images = int(250e3)

# =================== new directory ========================
for split in splits:

    for image_ext in image_exts1:
        file_path = data_path + split + 'rgb/' + '*' + '.png'
        print("File path: ", file_path)
        files = np.array(sorted(glob.glob(file_path)))

        mask_file_path = data_path + split + 'masks/' + '*' + '_label.png'
        print("File path: ", mask_file_path)
        mask_files = np.array(sorted(glob.glob(mask_file_path)))

        assert (len(files) == len(mask_files))
        print(f'Original dataset has {len(files)} examples')

        ### creating larger dataset
        if extend_dataset:
            rgb_idx, mask_idx = [], []
            total_idx = np.arange(0, len(files), 1)
            for image_idx in range(num_images):
                idx = np.random.choice(total_idx, size=1, replace=False)
                rgb_idx.append(files[int(idx)])
                mask_idx.append(mask_files[int(idx)])
            files, mask_files = rgb_idx, mask_idx
            assert (len(files) == len(mask_files))
        print(f'Extended dataset has {len(files)} examples')

        for rgb, mask in zip(files, mask_files):
            rgb_train_file.write(str(rgb))
            rgb_train_file.write("\n")
            labels_train_file.write(str(mask))
            labels_train_file.write("\n")

rgb_train_file.close()
labels_train_file.close()
