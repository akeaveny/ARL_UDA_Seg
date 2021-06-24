import numpy as np

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

affine_par = True

###########################
###########################

from pathlib import Path

ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config


###########################
###########################

def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


###########################
###########################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


###########################
###########################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


###########################
###########################

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


###########################
###########################

class ResNetMulti(nn.Module):
    def __init__(self, block, layers, num_classes, pretrained=True,
                 num_channels=3):
        self.num_channels = num_channels
        self.inplanes = 64
        super(ResNetMulti, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change

        ###########################
        # ResNet 101
        ###########################

        self.conv1 = nn.Conv2d(self.num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        ###########################
        # Classifier
        ###########################

        self.layer5_aux = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6_main = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()
        else:
            print("training from scratch .. ")

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, input, upsample=True):
        # ResNet Block 1
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # ResNet Block 2
        x = self.layer1(x)
        # ResNet Block 3
        x = self.layer2(x)
        # ResNet Block 4
        x = self.layer3(x)
        # ResNet Block 5
        x = self.layer4(x)

        # Classifier 1
        x1 = self.layer5(x)
        # Classifier 2
        x2 = self.layer6(x)

        ###########################
        ###########################
        if upsample:
            x1 = F.upsample(x1, size=input.size()[2:], mode='bilinear', align_corners=True)
            x2 = F.upsample(x2, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x1, x2

    ###########################
    ###########################

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        print("loading pre-trained weights .. {}".format(config.PRETRAINED_WEIGHTS))
        if config.PRETRAINED_WEIGHTS[:4] == 'http':
            # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
            saved_state_dict = model_zoo.load_url(config.PRETRAINED_WEIGHTS)
            new_params = self.state_dict().copy()
            #######################
            # D OR RBG+D INPUT
            #######################
            if config.NUM_CHANNELS != 3:
                print("not rgb input, pruning saved weights ..")
                pruned_saved_state_dict = {}
                for i in saved_state_dict:
                    i_parts = i.split('.')
                    if i_parts[1] != 'conv1' and i_parts[1] != 'bn1':
                        pruned_saved_state_dict[i] = saved_state_dict[i]
                saved_state_dict = pruned_saved_state_dict
            #######################
            # Classifier
            #######################
            for i in saved_state_dict:
                i_parts = i.split('.')
                if config.NUM_CLASSES == 21 or i_parts[1] != 'layer5':  ### 21 is from AdaptSegNet
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            self.load_state_dict(new_params)

    ###########################
    ###########################

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        # ResNet Block 1
        b.append(self.conv1)
        b.append(self.bn1)
        # ResNet Block 2
        b.append(self.layer1)
        # ResNet Block 3
        b.append(self.layer2)
        # ResNet Block 4
        b.append(self.layer3)
        # ResNet Block 5
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        # Classifier
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]

    ###########################
    ###########################


def DeeplabMulti(num_classes=config.NUM_CLASSES, pretrained=config.LOAD_PRETRAINED_WEIGHTS, num_channels=config.NUM_CHANNELS):
    model = ResNetMulti(Bottleneck, [3, 4, 23, 3], num_classes, pretrained=pretrained, num_channels=num_channels)
    return model


#########################
#########################

if __name__ == "__main__":
    model = DeeplabMulti(num_classes=config.NUM_CLASSES, pretrained=config.LOAD_PRETRAINED_WEIGHTS,
                    num_channels=config.NUM_CHANNELS)
    model.to(device=config.GPU)
    model.eval()

    ### print(model.optim_parameters(config.LEARNING_RATE))

    # from torchsummary import summary
    # TORCH_SUMMARY = (config.NUM_CHANNELS, 81, 46)
    # summary(model, TORCH_SUMMARY)

    image = torch.randn(config.BATCH_SIZE, config.NUM_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    print("\nImage: ", image.size())
    with torch.no_grad():
        pred_target_aux, pred_target_main = model.forward(image.to(device=config.GPU))
    print("pred_target_main:{}\npred_target_main:{}".format(pred_target_main.size(), pred_target_aux.size()))

