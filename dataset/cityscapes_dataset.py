import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image

class cityscapesDataSet(data.Dataset):
    def __init__(self, rgb_list, labels_list, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.rgb_list = rgb_list
        self.labels_list = labels_list
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(rgb_list)]
        self.label_ids = [i_id.strip() for i_id in open(labels_list)]
        assert(len(self.img_ids) == len(self.label_ids))
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for img, label in zip(self.img_ids, self.label_ids):
            self.files.append({
                "img": img,
                "label": label
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        ##############
        ##############

        image = Image.open(datafiles["img"]).convert('RGB')
        image = image.resize(self.crop_size, Image.BICUBIC)
        image = np.asarray(image, np.float32)

        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        ##############
        ##############
        label = Image.open(datafiles["label"])
        label = label.resize(self.crop_size, Image.NEAREST)
        label = np.asarray(label, np.float32)

        return image.copy(), label.copy()


if __name__ == '__main__':
    dst = cityscapesDataSet(rgb_list='/home/akeaveny/catkin_ws/src/AdaptSegNet/dataset/cityscapes_list/rgb_train_list.txt',
                            labels_list='/home/akeaveny/catkin_ws/src/AdaptSegNet/dataset/cityscapes_list/labels_train_list.txt')
    trainloader = data.DataLoader(dst, batch_size=1)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        ### img
        img = torchvision.utils.make_grid(imgs).numpy()
        img = np.transpose(np.array(img, dtype=np.int8), (1, 2, 0)) + dst.mean
        img = img[:, :, ::-1]
        ### label
        # label = torchvision.utils.make_grid(labels).numpy()
        label = np.array(labels, dtype=np.int8)
        label = np.resize(label, (321, 321))
        label
        ### plot
        plt.subplot(2,1,1)
        plt.imshow(img)
        plt.subplot(2,1,2)
        plt.imshow(label)
        plt.show()
        plt.show()
