# -*- coding: utf-8 -*-

import torch.optim as optim
import torchvision


# ---------- Print Tools ----------
def print_to_logfile(logfile, content, init=False, end='\n'):
    if init:
        with open(logfile, 'w') as f:
            f.write(content + end)
    else:
        with open(logfile, 'a') as f:
            f.write(content + end)


def print_to_console(content, colored=False):
    if not colored:
        print(content, flush=True)
    else:
        raise AssertionError('Not implemented yet')


# ---------- Data transform ----------
def make_transform(phase='train', output_size=448):
    if phase == 'train':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=output_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=output_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    elif phase == 'test':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=output_size),
            torchvision.transforms.CenterCrop(size=output_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        raise AssertionError('Not implemented yet')
    return transform


# ---------- Optimizer ----------
def make_optimizer(params, lr, weight_decay, opt='Adam'):
    if opt == 'Adam':
        print('| Adam Optimizer is used ... ')
        return optim.Adam(params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    elif opt == 'SGD':
        print('| SGD Optimizer is used ... ')
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    else:
        raise AssertionError('optimizer type is not implemented yet')


def get_lr_from_optimizer(optimizer):
    lr = -1
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr
