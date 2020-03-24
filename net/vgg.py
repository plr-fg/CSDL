# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class VGG16(nn.Module):
    def __init__(self, n_classes=200, pretrained=True):
        super().__init__()
        print('| A VGG16 network is instantiated, pre-trained: {}, number of classes: {}'.format(pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        vgg16 = torchvision.models.vgg16(pretrained=self._pretrained)
        self.features = vgg16.features
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.avgpool_embedding = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(*list(vgg16.classifier.children())[:-1])
        self.fc = nn.Linear(in_features=4096, out_features=self._n_classes)

        if self._pretrained:
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size(0)
        # assert x.size() == (N, 3, 448, 448)
        x = self.features(x)
        embedding = self.avgpool_embedding(x).view(N, -1)
        # assert x.size() == (N, 512, 14, 14)
        x = self.avgpool(x)
        x = x.view(N, -1)
        x = self.fc(self.classifier(x))
        assert x.size() == (N, self._n_classes)
        return embedding, x
