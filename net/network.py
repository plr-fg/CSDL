# -*- coding: utf-8 -*-

from net.vgg import VGG16
from net.googlenet import GoogLeNet
from net.resnet import ResNet50

def make_network(net):
    if net == 'vgg16':
        Net = VGG16
        feature_dim = 512
    elif net == 'resnet50':
        Net = ResNet50
        feature_dim = 2048
    elif net == 'googlenet':
        Net = GoogLeNet
        feature_dim = 1024
    else:
        raise AssertionError('please specify the correct net in {vgg16, resnet50, googlenet }')
    return Net, feature_dim
