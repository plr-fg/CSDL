# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


def smoothing_label_cross_entropy(logits, labels, epsilon=0.1, reduction='mean'):
    N = labels.size(0)
    C = logits.size(1)
    smoothed_label = torch.full(size=(N, C), fill_value=epsilon / (C - 1))
    smoothed_label.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1).cpu(), value=1-epsilon)
    if logits.is_cuda:
        smoothed_label = smoothed_label.cuda()
    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * smoothed_label, dim=1)  # (N)
    if reduction == 'none':
        return losses
    elif reduction == 'mean':
        return torch.sum(losses) / N
    elif reduction == 'sum':
        return torch.sum(losses)
    else:
        raise AssertionError('reduction has to be none, mean or sum')


class SmoothingLabelCrossEntropyLoss(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean'):
        super().__init__()
        self._epsilon = epsilon
        self._reduction = reduction
        print('> Using Smoothing Label Cross Entropy Loss, epsilon is {}'.format(self._epsilon))

    def forward(self, logits, labels):
        return smoothing_label_cross_entropy(logits, labels, self._epsilon, self._reduction)
