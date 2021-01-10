import numpy as np
import shutil
import glob
import os

# =================== new directory ========================
# 0.
data_path = '/data/Akeaveny/Datasets/domain_adaptation/ARLGAN/GTA5/'
list_path = '/home/akeaveny/catkin_ws/src/AdaptSegNet/dataset/gta5_list/'

rgb_train_file = open(list_path + 'rgb_train_list.txt', 'w')
labels_train_file = open(list_path + 'labels_train_list.txt', 'w')

splits = ['']

image_exts1 = ['']

# =================== new directory ========================
for split in splits:

    for image_ext in image_exts1:
        file_path = data_path + split + 'rgb/' + '*' + '.png'
        print("File path: ", file_path)
        files = np.array(sorted(glob.glob(file_path)))
        print("Loaded files: ", len(files))

        for idx, file in enumerate(files):
                rgb_train_file.write(str(file))
                rgb_train_file.write("\n")

    for image_ext in image_exts1:
        file_path = data_path + split + 'masks/' + '*' + '_label.png'
        print("File path: ", file_path)
        files = np.array(sorted(glob.glob(file_path)))
        print("Loaded files: ", len(files))

        ###############
        ###############

        for idx, file in enumerate(files):
                labels_train_file.write(str(file))
                labels_train_file.write("\n")

rgb_train_file.close()
labels_train_file.close()
