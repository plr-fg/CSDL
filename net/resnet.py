# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class ResNet50(nn.Module):
    """
    18: ([2, 2, 2, 2], basicblock),
    34: ([3, 4, 6, 3], basicblock),
    50: ([3, 4, 6, 3], bottleneck),
    101: ([3, 4, 23, 3], bottleneck),
    152: ([3, 8, 36, 3], bottleneck)
    """
    def __init__(self, n_classes=200, pretrained=True):
        super().__init__()
        print('| A ResNet50 network is instantiated, pre-trained: {}, '
              'number of classes: {}'.format(pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        resnet = torchvision.models.resnet50(pretrained=self._pretrained)
        # feature output is (N, 2048)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=2048, out_features=self._n_classes)

        if self._pretrained:
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        embedding = x.view(N, -1)
        x = self.fc(embedding)
        return embedding, x
