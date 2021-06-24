import torch

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)
ROOT_DIR_PATH = str(ROOT_DIR_PATH) + '/'

#######################################
#######################################

'''
FRAMEWORK Selection:
'Segmentation'
'SegmentationMulti'
'AdaptSegNet'
'CLAN'
'''

# TODO: prelim for naming
FRAMEWORK           = 'SegmentationMulti'
EXP_DATASET_NAME    = 'UMD_Real_RGBD_LF'
EXP_NUM             = 'v8_Deeplabv2'

#######################################
#######################################

'''
MODEL Selection:
'DeepLabDepthMulti'
'DeepLabv3'
'DeepLabv3Depth'
'DeepLabv3Multi'
'DeepLabv3DepthMulti'
'''

MODEL = 'DeepLabDepthMulti'
LAMBDA_SEG = 0.1               # 0.1 for AdaptSegNet with lower level features

IS_TRAIN_WITH_DEPTH = True
NUM_CHANNELS        = 4        # RGB=3, DEPTH=1 or RGB=4
NUM_RGB_CHANNELS    = 3 # NUM_CHANNELS - 1
NUM_D_CHANNELS      = 3 # NUM_CHANNELS - 3

CONFIDENCE_THRESHOLD = 0.35

LOAD_PRETRAINED_WEIGHTS = True
PRETRAINED_WEIGHTS = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

V3_OUTPUT_STRIDE = 16
V3_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

StartIteration     = 0
RESTORE_CHECKPOINT = None
BestFwb            = 0

#######################################
#######################################
''' saved weights '''

### AdaptSegNet
RESTORE_TRAINED_MODEL = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_TRAINED_MODEL_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_TRAINED_MODEL_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'

### CLAN
RESTORE_TRAINED_MODEL = '/home/akeaveny/catkin_ws/src/ARL_UDA_Seg/snapshots/CLAN_GTA5_82000.pth'

### Ours
RESTORE_TRAINED_MODEL = '/home/akeaveny/git/AdaptSegNet/snapshots/UMD_Real_RGBD_LF/SegmentationMulti_UMD_Real_RGBD_LF_128x128_v3_Low_Level_Feats/BEST_SEG_MODEL.pth'

#######################################
#######################################
''' segmentation configs '''

GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 1234

NUM_EPOCHS = 30
NUM_TRAIN  = 5000
NUM_VAL    = 1250

NUM_STEPS      = int(NUM_EPOCHS*NUM_TRAIN) # ~30 epochs at 5000 images/epoch
NUM_STEPS_STOP = NUM_STEPS

NUM_VAL_STEPS = int(NUM_EPOCHS*NUM_VAL)    # ~30 epochs at 1250 images/epoch

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 2

LEARNING_RATE = 2.5e-4
POWER = 0.9

# SGD
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

# ADAM
BETA_1 = 0.9
BETA_2 = 0.99

### IMG AUG
### AdaptSegNet
# IMG_MEAN = [104.00698793, 116.66876762, 122.67891434]             # ImageNet

### PR
PR_NUM_IMAGES = 28556 - 1 # zero index in files
# FS
PR_IMG_MEAN   = [138.58907803, 151.88310081, 125.22561575, 130.00560063]
PR_IMG_STD    = [30.37894422, 38.44065602, 43.2841762, 43.57943909]
# NORM
# PR_IMG_MEAN   = [138.58907803, 151.88310081, 125.22561575, 46.32026008]
# PR_IMG_STD    = [30.37894422, 38.44065602, 43.2841762, 16.97291491]
PR_RESIZE     = (int(640*1.35/3), int(480*1.35/3))
PR_INPUT_SIZE = (int(128), int(128))
### DR
# FS
DR_IMG_MEAN   = [134.38601217, 137.02879418, 129.27239013, 140.01491372]
DR_IMG_STD    = [48.88474747, 54.86081706, 48.8507932, 32.20115424]
# NORM
# DR_IMG_MEAN   = [134.38601217, 137.02879418, 129.27239013, 30.77622404]
# DR_IMG_STD    = [48.88474747, 54.86081706, 48.8507932, 7.461001]
DR_RESIZE     = (int(640/3), int(480/3))
DR_INPUT_SIZE = (int(128), int(128))
### SYN UMD
# FS
IMG_MEAN   = [135.4883242,  143.06856056, 125.6341276, 134.57706755]
IMG_STD    = [39.76640244, 46.91340711, 46.25064666, 38.62958981]
# NORM
# IMG_MEAN   = [135.4883242,  143.06856056, 125.6341276, 38.65048202]
# IMG_STD    = [39.76640244, 46.91340711, 46.25064666, 12.46277389]
RESIZE     = (int(640/3), int(480/3))
INPUT_SIZE = (int(128), int(128))

### REAL UMD
# FS
IMG_MEAN_TARGET   = [98.92739272, 66.78827961, 71.00867078, 135.8963934]
IMG_STD_TARGET    = [26.53540375, 31.51117582, 31.75977128, 38.23637208]
# NORM
# IMG_MEAN_TARGET   = [98.92739272, 66.78827961, 71.00867078, 31.36057707]
# IMG_STD_TARGET    = [26.53540375, 31.51117582, 31.75977128, 9.09040048]
RESIZE_TARGET     = (int(640/3), int(480/3))
INPUT_SIZE_TARGET = (int(128), int(128))

IMG_SIZE = str(INPUT_SIZE[0]) + 'x' + str(INPUT_SIZE[1])

NUM_TEST = 100
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"

#######################################
#######################################
''' GAN configs '''

LEARNING_RATE_D = 1e-4

################
# AdaptSegNet
################
GAN = 'Vanilla'
LAMBDA_ADV_TARGET_MAIN = 0.00025
LAMBDA_ADV_TARGET_AUX  = 0.00001

################
# CLAN
################
PREHEAT_STEPS = int(NUM_STEPS_STOP/20)

LAMBDA_CLAN_SEG = 1
LAMDA_WEIGHT = 0.01     # Weight Discrepancy Loss
LAMBDA_ADV = 0.00025      # Weighted BCE or Adaptive Adversarial Loss
LAMBDA_LOCAL = 40
EPSILON = 0.4

#######################################
#######################################
''' PRELIM FOR SAVED WEIGHTS'''

EVAL_UPDATE         = int(NUM_STEPS/150) # eval model every thousand iterations
TENSORBOARD_UPDATE  = int(NUM_STEPS/150)
SAVE_PRED_EVERY     = int(NUM_STEPS/NUM_EPOCHS*5)

EXP_NAME = FRAMEWORK + '_' + EXP_DATASET_NAME + '_' + IMG_SIZE + '_' + EXP_NUM
SNAPSHOT_DIR = str(ROOT_DIR_PATH) + 'snapshots/' + EXP_DATASET_NAME + '/' + EXP_NAME

MODEL_SAVE_PATH = str(SNAPSHOT_DIR) + '/'
BEST_MODEL_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_SEG_MODEL.pth'
BEST_DIS1_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_DIS1_MODEL.pth'
BEST_DIS2_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_DIS2_MODEL.pth'

#######################################
#######################################
''' DATASET PRELIMS'''

################
# Cityscapes
################

# ROOT_DATA_PATH = '/data/Akeaveny/Datasets/domain_adaptation/'
# USE_DEPTH_IMGS = False
# REMAP_LABEL = True
#
# NUM_CLASSES = 19
# IGNORE_LABEL = 255
#
# ### source
# DATA_DIRECTORY = ROOT_DATA_PATH + 'GTA5/'
# DATA_DIRECTORY_SOURCE_TRAIN = DATA_DIRECTORY
#
# ### target
# DATA_DIRECTORY_TARGET = ROOT_DATA_PATH + 'Cityscapes/'
# DATA_DIRECTORY_TARGET_TRAIN = DATA_DIRECTORY_TARGET + 'train/'
# DATA_DIRECTORY_TARGET_VAL = DATA_DIRECTORY_TARGET + 'val/'
# DATA_DIRECTORY_TARGET_TEST = DATA_DIRECTORY_TARGET + 'test/'
#
# ### TEST
# DATA_DIRECTORY_TARGET_TEST = ROOT_DATA_PATH + 'Cityscapes/val/pred1/'
# MIOU_TEST_INFO = ROOT_DIR_PATH + 'dataset/cityscapes_list/info.json'

################
# UMD
################

ROOT_DATA_PATH = '/data/Akeaveny/Datasets/domain_adaptation/UMD/'
REMAP_LABEL = False

NUM_CLASSES = 7 + 1         # 1 is for the background
IGNORE_LABEL = 255

### source
DATA_DIRECTORY = ROOT_DATA_PATH + 'Syn/'
DATA_DIRECTORY_SOURCE_TRAIN = DATA_DIRECTORY + 'train/'

DATA_DIRECTORY_PR = ROOT_DATA_PATH + 'PR/'
DATA_DIRECTORY_SOURCE_TRAIN_PR = DATA_DIRECTORY_PR + 'train/'

DATA_DIRECTORY_DR = ROOT_DATA_PATH + 'DR/'
DATA_DIRECTORY_SOURCE_TRAIN_DR = DATA_DIRECTORY_DR + 'train/'

### target
DATA_DIRECTORY_TARGET = ROOT_DATA_PATH + 'Real/'
DATA_DIRECTORY_TARGET_TRAIN = DATA_DIRECTORY_TARGET + 'train/'
DATA_DIRECTORY_TARGET_VAL = DATA_DIRECTORY_TARGET + 'val/'
DATA_DIRECTORY_TARGET_TEST = DATA_DIRECTORY_TARGET + 'test/'

### TEST
# TEST_SAVE_FOLDER = DATA_DIRECTORY_TARGET_TEST + 'pred/'
TEST_SAVE_FOLDER = DATA_DIRECTORY_TARGET_TEST + 'pred_' + EXP_NAME + '/'
MIOU_TEST_INFO = ROOT_DIR_PATH + 'dataset/cityscapes_list/info.json'

