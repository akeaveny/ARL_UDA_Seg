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
FRAMEWORK           = 'Segmentation'
EXP_DATASET_NAME    = 'UMD_Syn_RGBD_SE'
EXP_NUM             = 'v1'

#######################################
#######################################

'''
MODEL Selection:
'DeepLab'
'DeepLabDepth'
'DeepLabMulti'
'DeepLabv3'
'DeepLabv3Depth'
'DeepLabv3Multi'
'DeepLabv3DepthMulti'
'''

MODEL = 'DeepLabv3Depth'
LAMBDA_SEG = 1                 # 0.1 for AdaptSegNet with different classifiers
NUM_CHANNELS        = 4        # RGB=3 or DEPTH=1 or RGB=4
NUM_RGB_CHANNELS    = NUM_CHANNELS - 1
NUM_D_CHANNELS      = NUM_CHANNELS - 3

CONFIDENCE_THRESHOLD = 0.1

LOAD_PRETRAINED_WEIGHTS = True
PRETRAINED_WEIGHTS = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

V3_OUTPUT_STRIDE = 16
V3_PRETRAINED_WEIGHTS = 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'

StartIteration = 0
RESTORE_CHECKPOINT = None
BestFwb = None

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
# RESTORE_TRAINED_MODEL = '/home/akeaveny/git/ARL_UDA_Seg/trained_models/UMD/UMD_Real_RGB/Segmentation_UMD_Real_RGB_384x384_v0/BEST_SEG_MODEL.pth'
# RESTORE_TRAINED_MODEL = '/home/akeaveny/git/ARL_UDA_Seg/trained_models/UMD/UMD_Real_D/Segmentation_UMD_Real_D_384x384_v0/BEST_SEG_MODEL.pth'
# RESTORE_TRAINED_MODEL = '/home/akeaveny/git/ARL_UDA_Seg/trained_models/UMD/UMD_Real_RGBD_SE/Segmentation_UMD_Real_RGBD_SE_384x384_v0/BEST_SEG_MODEL.pth'

# RESTORE_TRAINED_MODEL = '/home/akeaveny/git/ARL_UDA_Seg/snapshots/UMD_Syn_RGBD_SE/Segmentation_UMD_Syn_RGBD_SE_480x480_v1/BEST_SEG_MODEL.pth'
RESTORE_TRAINED_MODEL = '/home/akeaveny/git/ARL_UDA_Seg/snapshots/UMD_Syn_RGBD_SE/Segmentation_UMD_Syn_RGBD_SE_384x384_v0/BEST_SEG_MODEL.pth'

#######################################
#######################################
''' segmentation configs '''

GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 1234

NUM_STEPS =      250000
NUM_STEPS_STOP = 150000 # early stopping

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

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
# IMG_MEAN = [104.00698793, 116.66876762, 122.67891434]             # AdaptSegNet

### SYN UMD
IMG_MEAN =   [147.1167, 127.1642, 128.9464, 162.40452575683594]
RESIZE =     (int(640*1.35), int(480*1.35))
INPUT_SIZE = (int(384*1), int(384*1))

### REAL UMD
IMG_MEAN_TARGET =   [94.2079, 73.0177, 71.1790, 135.5964]             # 384x384 UMD REAL RGB+D
RESIZE_TARGET =     (int(640*1), int(640*1))
INPUT_SIZE_TARGET = (int(384*1), int(384*1))

IMG_SIZE = str(INPUT_SIZE[0]) + 'x' + str(INPUT_SIZE[1])

NUM_TEST = 100
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"

#######################################
#######################################
''' GAN configs '''

################
# AdaptSegNet
################
GAN = 'Vanilla'

LEARNING_RATE_D = 1e-4
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

################
# CLAN
################
PREHEAT_STEPS = int(NUM_STEPS_STOP/20)

LAMDA_WEIGHT = 0.01
LAMBDA_ADV = 0.001
LAMBDA_LOCAL = 40
EPSILON = 0.4

#######################################
#######################################
''' PRELIM FOR SAVED WEIGHTS'''

EVAL_UPDATE         = int(NUM_STEPS/1000)
TENSORBOARD_UPDATE  = int(NUM_STEPS/1000)
SAVE_PRED_EVERY     = int(NUM_STEPS/25)

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
# ROOT_DATA_PATH = '/home/akeaveny/datasets/DomainAdaptation/UMD/'
USE_DEPTH_IMGS = True
REMAP_LABEL = False

NUM_CLASSES = 7 + 1         # 1 is for the background
IGNORE_LABEL = 255

### source
DATA_DIRECTORY = ROOT_DATA_PATH + 'Syn/'
DATA_DIRECTORY_SOURCE_TRAIN = DATA_DIRECTORY + 'train/'

### target
DATA_DIRECTORY_TARGET = ROOT_DATA_PATH + 'Real/'
DATA_DIRECTORY_TARGET_TRAIN = DATA_DIRECTORY_TARGET + 'train/'
DATA_DIRECTORY_TARGET_VAL = DATA_DIRECTORY_TARGET + 'val/'
DATA_DIRECTORY_TARGET_TEST = DATA_DIRECTORY_TARGET + 'test/'

### TEST
TEST_SAVE_FOLDER = DATA_DIRECTORY_TARGET + 'test/pred2/'
MIOU_TEST_INFO = ROOT_DIR_PATH + 'dataset/cityscapes_list/info.json'

