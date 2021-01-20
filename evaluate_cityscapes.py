import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
from packaging import version

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab import Res_Deeplab
from model.deeplab_multi import DeeplabMulti
from model.deeplab_vgg import DeeplabVGG
from dataset.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
from PIL import Image

import matplotlib.pyplot as plt
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/Cityscapes/data'
DATA_LIST_PATH = '/home/akeaveny/catkin_ws/src/AdaptSegNet/dataset/cityscapes_list/rgb_val_list.txt'
DATA_LABELS_PATH = '/home/akeaveny/catkin_ws/src/AdaptSegNet/dataset/cityscapes_list/labels_val_list.txt'

SAVE_PATH = '/data/Akeaveny/Datasets/domain_adaptation/ARLGAN/real/val/pred/'
gt_ext = "_gt.png"
pred_ext = "_pred.png"

# INPUT_SIZE = 1024,512
# INPUT_SIZE = 512, 256
INPUT_SIZE = 384, 384

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
# RESTORE_FROM = '/home/akeaveny/catkin_ws/src/AdaptSegNet/snapshots/AdaptSegNet_GTA2CITYSCAPES_0.5xResolution/GTA5_85000.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):

    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.model == 'DeeplabMulti':
        model = DeeplabMulti(num_classes=args.num_classes)
    elif args.model == 'Oracle':
        model = Res_Deeplab(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_ORC
    elif args.model == 'DeeplabVGG':
        model = DeeplabVGG(num_classes=args.num_classes)
        if args.restore_from == RESTORE_FROM:
            args.restore_from = RESTORE_FROM_VGG

    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    ### for running different versions of pytorch
    model_dict = model.state_dict()
    saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_dict}
    model_dict.update(saved_state_dict)
    ###
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda(gpu0)

    testloader = data.DataLoader(cityscapesDataSet(rgb_list=DATA_LIST_PATH, labels_list=DATA_LABELS_PATH,
                                                   crop_size=(INPUT_SIZE), mean=IMG_MEAN, scale=False, mirror=False),
                                    batch_size=1, shuffle=False, pin_memory=True)


    if version.parse(torch.__version__) >= version.parse('0.4.0'):
        interp = nn.Upsample(size=(INPUT_SIZE), mode='bilinear', align_corners=True)
    else:
        interp = nn.Upsample(size=(INPUT_SIZE), mode='bilinear')

    for index, batch in enumerate(testloader):
        print('%d processd' % index)
        # if index > 100:
        #     exit(0)
        image, label = batch
        if args.model == 'DeeplabMulti':
            output1, output2 = model(Variable(image, volatile=True).cuda(gpu0))
            # output = output2.cpu().data[0].numpy()
            output = interp(output2).cpu().data[0].numpy()
        elif args.model == 'DeeplabVGG' or args.model == 'Oracle':
            output = model(Variable(image, volatile=True).cuda(gpu0))
            # output = output.cpu().data[0].numpy()
            output = interp(output).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        ##################
        ### SAVE IMGS
        ##################

        label = Image.fromarray(np.squeeze(np.array(label, dtype=np.int8)))

        gt_name = SAVE_PATH + str(index) + gt_ext
        pred_name = SAVE_PATH + str(index) + pred_ext
        pred_color_name = SAVE_PATH + str(index) + '_color.png'

        label.save(gt_name)
        output.save(pred_name)
        output_col.save(pred_color_name)

        # import cv2
        # cv2.imshow("gt", np.array(np.array(label)*5, dtype=np.uint8)[:, :, np.newaxis])
        # cv2.imshow("pred", np.array(np.array(output_col), dtype=np.uint8)[:, :, np.newaxis])
        # cv2.waitKey(1)

        # plt.subplot(1, 2, 1)
        # plt.imshow(output)
        # plt.subplot(1, 2, 2)
        # plt.imshow(output_col)
        # plt.show()

if __name__ == '__main__':
    main()
