import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

###########################
###########################

from pathlib import Path
ROOT_DIR_PATH = Path(__file__).parents[1]

import cfg as config

#########################
#########################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,dilation=rate, padding=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.rate = rate

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

#########################
#########################

class ResNet(nn.Module):

    def __init__(self, nInputChannels, block, layers, os=16, pretrained=False):
        super(ResNet, self).__init__()
        if os == 16:
            strides = [1, 2, 2, 1]
            rates = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif os == 8:
            strides = [1, 2, 1, 1]
            rates = [1, 1, 2, 2]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #########################
        # RGB
        #########################

        self.inplanes = 64

        # ResNet Block 0
        self.conv1 = nn.Conv2d(nInputChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # ResNet Block 1
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        # ResNet Block 2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        # ResNet Block 3
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        # ResNet Block 4
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        #########################
        #########################

        self._init_weight()

        if pretrained:
            self._load_pretrained_model()
        else:
            print("training from scratch .. ")

    #########################
    #########################

    def _make_layer(self, block, planes, blocks, stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks=[1,2,4], stride=1, rate=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, rate=blocks[0]*rate, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1, rate=blocks[i]*rate))

        return nn.Sequential(*layers)

    #########################
    #########################

    def forward(self, rgb):
        # ResNet Block 0
        x_resnet0 = self.conv1(rgb)
        x_resnet0 = self.bn1(x_resnet0)
        x_resnet0 = self.relu(x_resnet0)
        x_resnet0 = self.maxpool(x_resnet0)

        # ResNet Block 1
        x_resnet1 = self.layer1(x_resnet0)
        low_level_feat = x_resnet1
        # ResNet Block 2
        x_resnet2 = self.layer2(x_resnet1)
        # ResNet Block 3
        x_resnet3 = self.layer3(x_resnet2)
        # ResNet Block 4
        x_resnet4 = self.layer4(x_resnet3)

        return x_resnet4, low_level_feat

    #########################
    #########################

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        print("loading pre-trained weights .. {}".format(config.V3_PRETRAINED_WEIGHTS))
        pretrain_dict = model_zoo.load_url(config.V3_PRETRAINED_WEIGHTS)
        model_dict = {}
        state_dict = self.state_dict()
        #######################
        # D OR RBG+D INPUT
        #######################
        if config.NUM_CHANNELS != 3:
            print("not rgb input, pruning saved weights ..")
            pruned_pretrain_dict = {}
            for i in pretrain_dict:
                i_parts = i.split('.')
                if i_parts[0] != 'conv1' and i_parts[0] != 'bn1':
                    pruned_pretrain_dict[i] = pretrain_dict[i]
            pretrain_dict = pruned_pretrain_dict
        #######################
        #######################
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

#########################
#########################

def ResNet101(nInputChannels=config.NUM_CHANNELS, os=config.V3_OUTPUT_STRIDE,
              pretrained=config.LOAD_PRETRAINED_WEIGHTS):
    model = ResNet(nInputChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
    return model

#########################
#########################

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate

        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                                            padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

#########################
#########################

class DeepLabv3Multi(nn.Module):
    def __init__(self, nInputChannels=config.NUM_CHANNELS,
                 n_classes=config.NUM_CLASSES,
                 os=config.V3_OUTPUT_STRIDE,
                 pretrained=config.LOAD_PRETRAINED_WEIGHTS,
                 _print=True):
        if _print:
            print("Constructing {} model...".format(config.MODEL))
            print("Number of RGB Channels: {}".format(nInputChannels))
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Batch Size: {}".format(config.BATCH_SIZE))
        super(DeepLabv3Multi, self).__init__()

        # ResNet 101
        self.resnet_features = ResNet101(nInputChannels, os, pretrained=pretrained)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.relu = nn.ReLU()

        #########################
        # Classifier 1
        #########################

        self.aspp1_c1 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2_c1 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3_c1 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4_c1 = ASPP_module(2048, 256, rate=rates[3])

        self.global_avg_pool_c1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU())

        self.conv1_c1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1_c1 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2_c1 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2_c1 = nn.BatchNorm2d(48)

        self.last_conv_c1 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        #########################
        # Classifier 2
        #########################

        self.aspp1_c2 = ASPP_module(2048, 256, rate=rates[0])
        self.aspp2_c2 = ASPP_module(2048, 256, rate=rates[1])
        self.aspp3_c2 = ASPP_module(2048, 256, rate=rates[2])
        self.aspp4_c2 = ASPP_module(2048, 256, rate=rates[3])

        self.global_avg_pool_c2 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                nn.Conv2d(2048, 256, 1, stride=1, bias=False),
                                                nn.BatchNorm2d(256),
                                                nn.ReLU())

        self.conv1_c2 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1_c2 = nn.BatchNorm2d(256)

        # adopt [1x1, 48] for channel reduction.
        self.conv2_c2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2_c2 = nn.BatchNorm2d(48)

        self.last_conv_c2 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          nn.BatchNorm2d(256),
                                          nn.ReLU(),
                                          nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    #########################
    #########################

    def forward(self, rgb):
        x, low_level_features = self.resnet_features(rgb)

        #########################
        # Classifier 1
        #########################
        ### ASPP
        x1 = self.aspp1_c1(x)
        x2 = self.aspp2_c1(x)
        x3 = self.aspp3_c1(x)
        x4 = self.aspp4_c1(x)
        self.global_avg_pool_c1.eval()
        x5 = self.global_avg_pool_c1(x)
        self.global_avg_pool_c1.train()

        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x_c1 = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x_c1 = self.conv1_c1(x_c1)
        x_c1 = self.bn1_c1(x_c1)
        x_c1 = self.relu(x_c1)
        x_c1 = F.upsample(x_c1, size=(int(math.ceil(rgb.size()[-2] // 4)),
                                      int(math.ceil(rgb.size()[-1] // 4))), mode='bilinear', align_corners=True)

        ### Low Level
        low_level_features_c1 = self.conv2_c1(low_level_features)
        low_level_features_c1 = self.bn2_c1(low_level_features_c1)
        low_level_features_c1 = self.relu(low_level_features_c1)

        ### Concat
        x_c1 = torch.cat((x_c1, low_level_features_c1), dim=1)
        x_c1 = self.last_conv_c1(x_c1)
        x_c1 = F.upsample(x_c1, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        #########################
        # Classifier 1
        #########################
        ### ASPP
        x1 = self.aspp1_c2(x)
        x2 = self.aspp2_c2(x)
        x3 = self.aspp3_c2(x)
        x4 = self.aspp4_c2(x)
        self.global_avg_pool_c2.eval()
        x5 = self.global_avg_pool_c2(x)
        self.global_avg_pool_c2.train()

        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x_c2 = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x_c2 = self.conv1_c2(x_c2)
        x_c2 = self.bn1_c2(x_c2)
        x_c2 = self.relu(x_c2)
        x_c2 = F.upsample(x_c2, size=(int(math.ceil(rgb.size()[-2] // 4)),
                                      int(math.ceil(rgb.size()[-1] // 4))), mode='bilinear', align_corners=True)

        ### Low Level
        low_level_features_c2 = self.conv2_c1(low_level_features)
        low_level_features_c2 = self.bn2_c1(low_level_features_c2)
        low_level_features_c2 = self.relu(low_level_features_c2)

        ### Concat
        x_c2 = torch.cat((x_c2, low_level_features_c2), dim=1)
        x_c2 = self.last_conv_c2(x_c2)
        x_c2 = F.upsample(x_c2, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        #########################
        #########################

        return x_c1, x_c2

    #########################
    #########################

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #########################
    #########################

    def get_1x_lr_params(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = [self.resnet_features]
        for i in range(len(b)):
            for k in b[i].parameters():
                if k.requires_grad:
                    yield k


    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [
            self.aspp1_c1, self.aspp2_c1, self.aspp3_c1, self.aspp4_c1,
            self.conv1_c1, self.conv2_c1, self.last_conv_c1,
            self.aspp1_c2, self.aspp2_c2, self.aspp3_c2, self.aspp4_c2,
            self.conv1_c2, self.conv2_c2, self.last_conv_c2
           ]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def get_classifier_1_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [
            self.aspp1_c1, self.aspp2_c1, self.aspp3_c1, self.aspp4_c1,
            self.conv1_c1, self.conv2_c1, self.last_conv_c1,
           ]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def get_classifier_2_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = [
            self.aspp1_c2, self.aspp2_c2, self.aspp3_c2, self.aspp4_c2,
            self.conv1_c2, self.conv2_c2, self.last_conv_c2
           ]
        for j in range(len(b)):
            for k in b[j].parameters():
                if k.requires_grad:
                    yield k

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params(), 'lr': learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * learning_rate}]

#########################
#########################

if __name__ == "__main__":
    model = DeepLabv3Multi(nInputChannels=config.NUM_CHANNELS, n_classes=config.NUM_CLASSES,
                           os=config.V3_OUTPUT_STRIDE,
                           pretrained=config.LOAD_PRETRAINED_WEIGHTS,
                           _print=True)
    model.to(device=config.GPU)
    model.eval()

    ### print(model.optim_parameters(config.LEARNING_RATE))

    # from torchsummary import summary
    # TORCH_SUMMARY = (config.NUM_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    # summary(model, TORCH_SUMMARY)

    image = torch.randn(config.BATCH_SIZE, config.NUM_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    print("\nImage: ", image.size())
    with torch.no_grad():
        pred_target_aux, pred_target_main = model.forward(image.to(device=config.GPU))
    print("pred_target_main:{}\npred_target_main:{}".format(pred_target_main.size(), pred_target_aux.size()))