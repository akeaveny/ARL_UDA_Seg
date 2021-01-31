import torch

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parent.absolute().resolve(strict=True)
ROOT_DIR_PATH = str(ROOT_DIR_PATH) + '/'

ROOT_DATA_PATH = '/home/akeaveny/datasets/ARLGAN/'

#######################################
#######################################

'''
Model Selection:
'DeepLabMulti'
'''

MODEL = 'DeepLabMulti'

LOAD_PRETRAINED_WEIGHTS = True
PRETRAINED_WEIGHTS = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'

StartIteration = 0
RESTORE_CHECKPOINT = None

RESTORE_TRAINED_MODEL = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_TRAINED_MODEL_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_TRAINED_MODEL_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'

# RESTORE_TRAINED_MODEL = '/home/akeaveny/git/AdaptSegNet/snapshots/GTA5/TEST/final_saved_model_12.82_mIoU.pth'

#######################################
#######################################

RANDOM_SEED = 1234

NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
EXTEND_DATASET = True

BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4

LEARNING_RATE = 2.5e-4
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

POWER = 0.9

### IMG AUG
IMG_MEAN = [104.00698793, 116.66876762, 122.67891434]

#######################################
### GAN
#######################################

TARGET = 'cityscapes'

GAN = 'Vanilla'

LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001

#######################################
# PRELIM FOR SAVED WEIGHTS
#######################################

GPU = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EVAL_UPDATE         = int(NUM_STEPS_STOP/100)
TENSORBOARD_UPDATE  = int(NUM_STEPS_STOP/1000)

# prelim for naming
FRAMEWORK       = 'AdaptSegNet'
TRAINING_MODE   = 'DIS'                      # 'SEGMENTATION' or 'DIS'
DATASET_NAME    = 'GTA5_2_CITYSCAPES'        # 'GTA5' or 'GTA5_2_CITYSCAPES
IMG_SIZE        = '640x360'
EXP_NUM         = 'v0'

EXP_NAME = FRAMEWORK + '_' + TRAINING_MODE + '_' + DATASET_NAME + '_' + IMG_SIZE + '_' + EXP_NUM
SNAPSHOT_DIR = str(ROOT_DIR_PATH) + 'snapshots/' + DATASET_NAME + '/' + EXP_NAME

MODEL_SAVE_PATH = str(SNAPSHOT_DIR) + '/'
BEST_MODEL_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_SEG_MODEL.pth'
BEST_DIS1_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_DIS1_MODEL.pth'
BEST_DIS2_SAVE_PATH = MODEL_SAVE_PATH + 'BEST_DIS2_MODEL.pth'

#######################################
# DATASET PRELIMS
#######################################

NUM_CLASSES = 19
IGNORE_LABEL = 255

### GTA5
# INPUT_SIZE = (1280,720)
INPUT_SIZE = (640, 360)

### CITYSCAPES
# INPUT_SIZE_TARGET = (1024,512)
INPUT_SIZE_TARGET = (512, 256)

TORCH_SUMMARY = (3, 81, 46)         # from testing with deeplab_multi.py

#######################################
# GTA5
#######################################

### source

DATA_DIRECTORY = ROOT_DATA_PATH + 'GTA5/'
DATA_DIRECTORY_SOURCE_TRAIN = DATA_DIRECTORY

# DATA_LIST_PATH = ROOT_DIR_PATH + 'dataset/gta5_list/rgb_train_list.txt'
# DATA_LABEL_PATH = ROOT_DIR_PATH + 'dataset/gta5_list/labels_train_list.txt'

#######################################
# Cityscapes
#######################################

### target
DATA_DIRECTORY_TARGET = ROOT_DATA_PATH + 'Cityscapes/'
DATA_DIRECTORY_TARGET_TRAIN = DATA_DIRECTORY_TARGET + 'train/'
DATA_DIRECTORY_TARGET_VAL = DATA_DIRECTORY_TARGET + 'val/'
DATA_DIRECTORY_TARGET_TEST = DATA_DIRECTORY_TARGET + 'test/'

# DATA_LIST_PATH_TARGET = ROOT_DIR_PATH + 'dataset/cityscapes_list/rgb_train_list.txt'
# DATA_LABELS_PATH_TARGET = ROOT_DIR_PATH + 'dataset/cityscapes_list/labels_train_list.txt'

### TEST
NUM_TEST = 25
TEST_SAVE_PATH = ROOT_DATA_PATH + 'Cityscapes/val/pred/'
TEST_GT_EXT = "_gt.png"
TEST_PRED_EXT = "_pred.png"

MIOU_TEST_INFO = ROOT_DIR_PATH + 'dataset/cityscapes_list/info.json'
# DATA_LIST_PATH_TEST = ROOT_DIR_PATH + 'dataset/cityscapes_list/rgb_val_list.txt'
# DATA_LABELS_PATH_TEST = ROOT_DIR_PATH + 'dataset/cityscapes_list/labels_val_list.txt'

