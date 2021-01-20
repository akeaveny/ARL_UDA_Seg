import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import glob
import matplotlib.pyplot as plt

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(labels_dir, json_file):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(json_file, 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    gt_path = labels_dir + '*' + '_gt.png'
    pred_path = labels_dir + '*' + '_pred.png'
    gt_files = np.array(sorted(glob.glob(gt_path)))
    pred_files = np.array(sorted(glob.glob(pred_path)))
    assert(len(gt_files) == len(pred_files))
    print("Loaded files: ", len(gt_files))

    for ind, files in enumerate(zip(gt_files, pred_files)):
        gt_file, pred_file = files[0], files[1]
        pred = np.array(Image.open(pred_file))
        label = np.array(Image.open(gt_file))
        label = label_mapping(label, mapping)

        # plt.subplot(2, 1, 1)
        # plt.imshow(np.array(label))
        # plt.subplot(2, 1, 2)
        # plt.imshow(np.array(pred))
        # plt.show()
        # plt.show()

        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_file, pred_file))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_files), 100*np.mean(per_class_iu(hist))))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs


def main(args):
   compute_mIoU(args.labels_dir, args.json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir', default='/data/Akeaveny/Datasets/domain_adaptation/ARLGAN/real/val/pred/')
    parser.add_argument('--json_file', default='/home/akeaveny/catkin_ws/src/AdaptSegNet/dataset/cityscapes_list/info.json')
    args = parser.parse_args()
    main(args)
