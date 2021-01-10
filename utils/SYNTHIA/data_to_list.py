import numpy as np
import shutil
import glob
import os

# =================== new directory ========================
# 0.
data_path = '/data/Akeaveny/Datasets/domain_adaptation/ARLGAN/real/'
list_path = '/home/akeaveny/catkin_ws/src/AdaptSegNet/dataset/cityscapes_list/'

rgb_train_file = open(list_path + 'rgb_train_list.txt', 'w')
labels_train_file = open(list_path + 'labels_train_list.txt', 'w')

rgb_val_file = open(list_path + 'rgb_val_list.txt', 'w')
labels_val_file = open(list_path + 'labels_val_list.txt', 'w')

rgb_test_file = open(list_path + 'rgb_test_list.txt', 'w')
labels_test_file = open(list_path + 'labels_test_list.txt', 'w')

splits = [
          'train/',
          'val/',
          'test/'
]

image_exts1 = [
    '.png',
    '_label.png'
]

# =================== new directory ========================
for split in splits:
    offset = 0
    for image_ext in image_exts1:
        file_path = data_path + split + '*' + image_ext
        print("File path: ", file_path)
        files = np.array(sorted(glob.glob(file_path)))
        print("Loaded files: ", len(files))

        ###############
        #
        ###############

        for idx, file in enumerate(files):

            if image_ext == '.png':
                if split == 'train/':
                    rgb_train_file.write(str(file))
                    rgb_train_file.write("\n")
                elif split == 'val/':
                    rgb_val_file.write(str(file))
                    rgb_val_file.write("\n")
                elif split == 'test/':
                    rgb_test_file.write(str(file))
                    rgb_test_file.write("\n")

            elif image_ext == '_label.png':
                if split == 'train/':
                    labels_train_file.write(str(file))
                    labels_train_file.write("\n")
                elif split == 'val/':
                    labels_val_file.write(str(file))
                    labels_val_file.write("\n")
                elif split == 'test/':
                    labels_test_file.write(str(file))
                    labels_test_file.write("\n")

rgb_train_file.close(), rgb_val_file.close(), rgb_test_file.close()
labels_train_file.close(), labels_val_file.close(), labels_test_file.close()
