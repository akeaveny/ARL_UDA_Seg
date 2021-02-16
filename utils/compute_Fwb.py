import argparse
from PIL import Image
import numpy as np
import sys
import os
import glob

import scipy
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import distance_transform_edt

import matplotlib.pyplot as plt

######################
######################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1].absolute().resolve(strict=True)

import cfg as config

######################
######################

def matlab_style_gauss2D(shape=(7,7),sigma=5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def Wfb(foreground, groundtruth):

    foreground = np.array(foreground, dtype=float)
    groundtruth = np.array(groundtruth, dtype=float)

    dist, dist_idx = distance_transform_edt(1 - groundtruth, return_distances=True, return_indices=True)

    error = abs(foreground - groundtruth)
    error_t = error
    min_error_ea = error

    groundtruth_bool = np.array(groundtruth, dtype=bool)
    # error_t[~groundtruth_bool] = error_t[dist_idx[1, ...][~groundtruth_bool], -1]
    error_t[~groundtruth_bool] = error_t[-1, dist_idx[1, ...][~groundtruth_bool]]

    ### k = matlab_style_gauss2D(shape=(7,7), sigma=5)
    error_abs = gaussian_filter(error_t, sigma=5)

    error_bool = np.array(error, dtype=bool)
    error_abs_bool = np.array(error_abs, dtype=bool)
    min_error_ea[groundtruth_bool & error_abs_bool < error_bool] = error_abs[groundtruth_bool & error_abs_bool < error_bool]

    background = np.ones(shape=groundtruth.shape)
    background[~groundtruth_bool] = 2 - 1 * np.exp(np.log(1 - 0.5) / 5 * dist[~groundtruth_bool])
    error_w = min_error_ea * background

    TPw = sum(sum(groundtruth)) - sum(error_w[groundtruth_bool])
    FPw = sum(error_w[~groundtruth_bool])

    WeighedRecall = 1 - np.mean(error_w[groundtruth_bool])
    WeighedPrecision = TPw / (np.spacing(1) + TPw + FPw)

    return (2 * WeighedRecall * WeighedPrecision) / (np.spacing(1) + WeighedRecall + WeighedPrecision)

def weightedFb(path=config.TEST_SAVE_FOLDER, aff_start=0, aff_end=config.NUM_CLASSES,
               verbose=False, visualize=False):

    affordances = np.arange(aff_start+1, aff_end) # +1 ignores the background

    #################
    #################

    image_path_gt = path + '*' + config.TEST_GT_EXT
    list_gt = sorted(glob.glob(image_path_gt))

    image_path_pred = path + '*' + config.TEST_PRED_EXT
    list_pred = sorted(glob.glob(image_path_pred))

    num_gt, num_pred = len(list_gt), len(list_pred)
    assert num_gt == num_pred

    f_wb_affordance = np.zeros(num_gt)
    f_wb_rank = []

    #################
    #################
    if verbose:
        print()

    for affordance_idx, affordance_id in enumerate(affordances):
        for image_idx, addr in enumerate(zip(list_gt, list_pred)):

            gt_addr, pred_addr = addr[0], addr[1]
            gt_img = np.array(Image.open('{0}'.format(gt_addr)))
            pred_img = np.array(Image.open('{0}'.format(pred_addr)))

            if visualize:
                #################
                plt.figure(0)
                plt.subplot(1, 2, 1)
                plt.title("GT")
                plt.imshow(gt_img)
                plt.subplot(1, 2, 2)
                plt.title("Pred")
                plt.imshow(np.array(pred_img))
                plt.show()
                plt.ioff()
                #################

            foreground = pred_img == affordance_id
            groundtruth = gt_img == affordance_id

            if sum(sum(groundtruth)) > 0:
                f_wb_affordance[image_idx] = Wfb(foreground, groundtruth)

        f_wb_rank.append(np.mean(f_wb_affordance))
        if verbose:
            print('Averaged F_wb for affordance id={} is: {:.3f}'.format(affordance_id, f_wb_rank[-1]))
    fwb = np.mean(f_wb_rank)
    if verbose:
        print('\n*** Averaged F_wb for all Affordances is: {:.3f} ***\n'.format(fwb))
    return fwb

if __name__ == '__main__':
    weightedFb(verbose=False, visualize=False)