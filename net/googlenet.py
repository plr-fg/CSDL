# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision


torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


class GoogLeNet(nn.Module):
    def __init__(self, n_classes=200, pretrained=True):
        super().__init__()
        print('| A GoogLeNet network is instantiated, pre-trained: {}, '
              'number of classes: {}'.format(pretrained, n_classes))

        self._pretrained = pretrained
        self._n_classes = n_classes
        googlenet = torchvision.models.googlenet(pretrained=self._pretrained, aux_logits=False)

        self.features = nn.Sequential(*list(googlenet.children())[:-3])
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(in_features=1024, out_features=self._n_classes)

        if self._pretrained:
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                nn.init.constant_(self.fc.bias.data, val=0)

    def forward(self, x):
        N = x.size(0)
        x = self.features(x)
        x = self.avgpool(x)
        embedding = x.view(N, -1)
        x = self.dropout(x)
        x = x.view(N, -1)
        x = self.fc(x)
        return embedding, x
