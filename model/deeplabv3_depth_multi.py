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

from utils.helper_models import SqueezeAndExciteFusionAdd

#########################
#########################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, rate=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=rate, padding=rate, bias=False)
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

class ResNetDepth(nn.Module):

    def __init__(self, nRGBChannels, nDChannels, block, layers, os=16, pretrained=False):
        super(ResNetDepth, self).__init__()
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
        self.conv1 = nn.Conv2d(nRGBChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
        # DEPTH
        #########################

        self.inplanes = 64

        # ResNet Block 0
        self.conv1_depth = nn.Conv2d(nDChannels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1_depth = nn.BatchNorm2d(64)

        # ResNet Block 1
        self.layer1_depth = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
        # ResNet Block 2
        self.layer2_depth = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
        # ResNet Block 3
        self.layer3_depth = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
        # ResNet Block 4
        self.layer4_depth = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], rate=rates[3])

        ###########################
        # Fusion
        ###########################

        self.se_resnet_0 = SqueezeAndExciteFusionAdd(64, activation=self.relu)
        self.se_resnet_1 = SqueezeAndExciteFusionAdd(256, activation=self.relu)
        self.se_resnet_2 = SqueezeAndExciteFusionAdd(512, activation=self.relu)
        self.se_resnet_3 = SqueezeAndExciteFusionAdd(1024, activation=self.relu)
        self.se_resnet_4 = SqueezeAndExciteFusionAdd(2048, activation=self.relu)

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

    def forward(self, rgb, depth):

        ###########################
        # depth
        ###########################
        # ResNet Block 1
        x_depth_resnet0 = self.conv1_depth(depth)
        x_depth_resnet0 = self.bn1_depth(x_depth_resnet0)
        x_depth_resnet0 = self.relu(x_depth_resnet0)
        x_depth_resnet0 = self.maxpool(x_depth_resnet0)

        # ResNet Block 2
        x_depth_resnet1 = self.layer1_depth(x_depth_resnet0)
        low_level_depth_feat = x_depth_resnet1
        # ResNet Block 3
        x_depth_resnet2 = self.layer2_depth(x_depth_resnet1)
        # ResNet Block 4
        x_depth_resnet3 = self.layer3_depth(x_depth_resnet2)
        # ResNet Block 5
        x_depth_resnet4 = self.layer4_depth(x_depth_resnet3)

        ###########################
        # RGB
        ###########################

        # ResNet Block 0
        x_resnet0 = self.conv1(rgb)
        x_resnet0 = self.bn1(x_resnet0)
        x_resnet0 = self.relu(x_resnet0)
        x_resnet0 = self.maxpool(x_resnet0)
        # x_resnet0 = self.se_resnet_0(rgb=x_resnet0, depth=x_depth_resnet0)

        # ResNet Block 1
        x_resnet1 = self.layer1(x_resnet0)
        # x_resnet1 = self.se_resnet_1(rgb=x_resnet1, depth=x_depth_resnet1)
        low_level_rgb_feat = x_resnet1
        # ResNet Block 2
        x_resnet2 = self.layer2(x_resnet1)
        # x_resnet2 = self.se_resnet_2(rgb=x_resnet2, depth=x_depth_resnet2)
        # ResNet Block 3
        x_resnet3 = self.layer3(x_resnet2)
        # x_resnet3 = self.se_resnet_3(rgb=x_resnet3, depth=x_depth_resnet3)
        # ResNet Block 4
        x_resnet4 = self.layer4(x_resnet3)
        # x_resnet4 = self.se_resnet_4(rgb=x_resnet4, depth=x_depth_resnet4)

        ###########################
        ### TODO: Addition or Concat (see paper ..)
        ###########################

        # low_level_feat = torch.cat((low_level_rgb_feat, low_level_depth_feat), dim=1)
        x_resnet3 = torch.cat((x_resnet3, x_depth_resnet3), dim=1)
        x_resnet4 = torch.cat((x_resnet4, x_depth_resnet4), dim=1)

        # x_resnet4, x_resnet3, low_level_feat
        return x_resnet4, x_resnet3, low_level_rgb_feat

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
        if config.NUM_CHANNELS == 1:
            print("not rgb input, pruning saved weights ..")
            pruned_pretrain_dict = {}
            for i in pretrain_dict:
                i_parts = i.split('.')
                if i_parts[0] != 'conv1' and i_parts[0] != 'bn1' and i_parts[0] != 'layer1': # layer1 for depth branch
                    pruned_pretrain_dict[i] = pretrain_dict[i]
            pretrain_dict = pruned_pretrain_dict
        #######################
        #######################
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
                ### for depth
                _parts = k.split('.')
                _depth_name = _parts[0] + '_depth.' + '.'.join(_parts[1:])
                model_dict[_depth_name] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

#########################
#########################

def ResNetDepth101(nRGBChannels=config.NUM_CHANNELS, nDChannels=config.NUM_D_CHANNELS,
              os=config.V3_OUTPUT_STRIDE,
              pretrained=config.LOAD_PRETRAINED_WEIGHTS):
    model = ResNetDepth(nRGBChannels, nDChannels, Bottleneck, [3, 4, 23, 3], os, pretrained=pretrained)
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

class DeepLabv3DepthMulti(nn.Module):
    def __init__(self, nRGBChannels=config.NUM_RGB_CHANNELS, nDChannels=config.NUM_D_CHANNELS,
                 n_classes=config.NUM_CLASSES,
                 os=config.V3_OUTPUT_STRIDE,
                 pretrained=config.LOAD_PRETRAINED_WEIGHTS,
                 _print=True):
        if _print:
            print("Constructing {} model...".format(config.MODEL))
            print("Number of RGB Channels: {}".format(nRGBChannels))
            print("Number of Depth Channels: {}".format(nDChannels))
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Batch Size: {}".format(config.BATCH_SIZE))
        super(DeepLabv3DepthMulti, self).__init__()

        # ResNet 101
        self.resnet_rgbd_features = ResNetDepth101(nRGBChannels, nDChannels, os, pretrained=pretrained)
        self.main_features  = int(2048*2)
        self.aux_features   = int(1024*2)
        self.low_level_feat = int(256*1)

        # ASPP
        if os == 16:
            rates = [1, 6, 12, 18]
        elif os == 8:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.relu = nn.ReLU()

        #########################
        # Classifier 1: main
        #########################

        ## RGB
        self.aspp1_main_rgb = ASPP_module(self.main_features, 256, rate=rates[0])
        self.aspp2_main_rgb = ASPP_module(self.main_features, 256, rate=rates[1])
        self.aspp3_main_rgb = ASPP_module(self.main_features, 256, rate=rates[2])
        self.aspp4_main_rgb = ASPP_module(self.main_features, 256, rate=rates[3])

        self.global_avg_pool_main_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                  nn.Conv2d(self.main_features, 256, 1, stride=1, bias=False),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU())

        self.conv1_main_rgb = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1_main_rgb = nn.BatchNorm2d(256)

        self.conv2_main_rgb = nn.Conv2d(self.low_level_feat, 48, 1, bias=False)
        self.bn2_main_rgb = nn.BatchNorm2d(48)

        ###
        self.last_conv_main = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),
                                            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                            nn.BatchNorm2d(256),
                                            nn.ReLU(),
                                            nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

        #########################
        # Classifier 2: aux
        #########################

        ### RGB
        self.aspp1_aux_rgb = ASPP_module(self.aux_features, 256, rate=rates[0])
        self.aspp2_aux_rgb = ASPP_module(self.aux_features, 256, rate=rates[1])
        self.aspp3_aux_rgb = ASPP_module(self.aux_features, 256, rate=rates[2])
        self.aspp4_aux_rgb = ASPP_module(self.aux_features, 256, rate=rates[3])

        self.global_avg_pool_aux_rgb = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(self.aux_features, 256, 1, stride=1, bias=False),
                                                 nn.BatchNorm2d(256),
                                                 nn.ReLU())

        self.conv1_aux_rgb = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1_aux_rgb = nn.BatchNorm2d(256)

        self.conv2_aux_rgb = nn.Conv2d(self.low_level_feat, 48, 1, bias=False)
        self.bn2_aux_rgb = nn.BatchNorm2d(48)

        ###
        self.last_conv_aux = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(),
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(),
                                           nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    #########################
    #########################

    def forward(self, rgb, depth):
        # x_resnet4, x_resnet3, low_level_feat
        x_resnet4, x_resnet3, resnet_low_level_features = self.resnet_rgbd_features(rgb, depth)

        #########################
        # Classifier 1: main
        #########################

        ### RGB
        x1 = self.aspp1_main_rgb(x_resnet4)
        x2 = self.aspp2_main_rgb(x_resnet4)
        x3 = self.aspp3_main_rgb(x_resnet4)
        x4 = self.aspp4_main_rgb(x_resnet4)

        self.global_avg_pool_main_rgb.eval()
        x5 = self.global_avg_pool_main_rgb(x_resnet4)
        self.global_avg_pool_main_rgb.train()

        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1_main_rgb(x)
        x = self.bn1_main_rgb(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(rgb.size()[-2] // 4)),
                                int(math.ceil(rgb.size()[-1] // 4))), mode='bilinear', align_corners=True)

        ###
        low_level_features = self.conv2_main_rgb(resnet_low_level_features)
        low_level_features = self.bn2_main_rgb(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)

        ###
        x = self.last_conv_main(x)
        x_main = F.upsample(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        #########################
        # Classifier 2: aux
        #########################

        ### RGB
        x1 = self.aspp1_aux_rgb(x_resnet3)
        x2 = self.aspp2_aux_rgb(x_resnet3)
        x3 = self.aspp3_aux_rgb(x_resnet3)
        x4 = self.aspp4_aux_rgb(x_resnet3)

        self.global_avg_pool_aux_rgb.eval()
        x5 = self.global_avg_pool_aux_rgb(x_resnet3)
        self.global_avg_pool_aux_rgb.train()

        x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1_aux_rgb(x)
        x = self.bn1_aux_rgb(x)
        x = self.relu(x)
        x = F.upsample(x, size=(int(math.ceil(rgb.size()[-2] // 4)),
                                int(math.ceil(rgb.size()[-1] // 4))), mode='bilinear', align_corners=True)

        ###
        low_level_features = self.conv2_aux_rgb(resnet_low_level_features)
        low_level_features = self.bn2_aux_rgb(low_level_features)
        low_level_features = self.relu(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)

        ###
        x = self.last_conv_aux(x)
        x_aux = F.upsample(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        #########################
        #########################

        # pred_target_aux, pred_target_main
        return x_aux, x_main

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
        b = [self.resnet_rgbd_features]
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
            # Main: RGB
            self.aspp1_main_rgb, self.aspp2_main_rgb, self.aspp3_main_rgb, self.aspp4_main_rgb,
            self.conv1_main_rgb, self.conv2_main_rgb, self.last_conv_main,
            # AUX: RGB
            self.aspp1_aux_rgb, self.aspp2_aux_rgb, self.aspp3_aux_rgb, self.aspp4_aux_rgb,
            self.conv1_aux_rgb, self.conv2_aux_rgb, self.last_conv_aux,
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
            # Main: RGB
            self.aspp1_main_rgb, self.aspp2_main_rgb, self.aspp3_main_rgb, self.aspp4_main_rgb,
            self.conv1_main_rgb, self.conv2_main_rgb, self.last_conv_main,
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
            # AUX: RGB
            self.aspp1_aux_rgb, self.aspp2_aux_rgb, self.aspp3_aux_rgb, self.aspp4_aux_rgb,
            self.conv1_aux_rgb, self.conv2_aux_rgb, self.last_conv_aux,
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
    model = DeepLabv3DepthMulti(nRGBChannels=config.NUM_RGB_CHANNELS, nDChannels=config.NUM_D_CHANNELS,
                           n_classes=config.NUM_CLASSES,
                           os=config.V3_OUTPUT_STRIDE,
                           pretrained=config.LOAD_PRETRAINED_WEIGHTS,
                           _print=True)
    model.to(device=config.GPU)
    model.eval()
    model.get_1x_lr_params()

    ### print(model.optim_parameters(config.LEARNING_RATE))

    # from torchsummary import summary
    # TORCH_SUMMARY = (config.NUM_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    # summary(model, TORCH_SUMMARY)

    image = torch.randn(config.BATCH_SIZE, config.NUM_RGB_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    depth = torch.randn(config.BATCH_SIZE, config.NUM_D_CHANNELS, config.INPUT_SIZE[0], config.INPUT_SIZE[1])
    print("\nImage:{}\nDepth:{}".format(image.size(), depth.size()))
    with torch.no_grad():
        pred_target_aux, pred_target_main = model.forward(image.to(device=config.GPU), depth.to(device=config.GPU))
    print("pred_target_main:{}\npred_target_main:{}".format(pred_target_main.size(), pred_target_aux.size()))