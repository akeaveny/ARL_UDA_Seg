import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

###########################
###########################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

###########################
###########################

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, segmap, upsample=True):
        # 1st
        x = self.conv1(segmap)
        x = self.leaky_relu(x)
        # 2nd
        x = self.conv2(x)
        x = self.leaky_relu(x)
        # 3rd
        x = self.conv3(x)
        x = self.leaky_relu(x)
        # 4th
        x = self.conv4(x)
        x = self.leaky_relu(x)
        # classifier
        x = self.classifier(x)

        ###########################
        ###########################
		# if upsample:
		# 	x = F.upsample(x, size=segmap.size()[2:], mode='bilinear', align_corners=True)

        return x

#########################
#########################

if __name__ == "__main__":
    model = FCDiscriminator(num_classes=config.NUM_CLASSES)
    model.to(device=config.GPU)
    model.eval()

    upsample_source = nn.Upsample(size=config.INPUT_SIZE, mode='bilinear', align_corners=True)
    upsample_target = nn.Upsample(size=config.INPUT_SIZE_TARGET, mode='bilinear', align_corners=True)

    image = torch.randn(config.BATCH_SIZE, config.NUM_CLASSES, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    print("\nImage: ", image.size())
    with torch.no_grad():
        pred_target_aux = upsample_source(model.forward(image.to(device=config.GPU)))
    print("pred_target_aux: {}".format(pred_target_aux.size()))
