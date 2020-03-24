# -*- coding: utf-8 -*-

import torchvision


def make_transform_cifar(phase='train'):
    if phase == 'train':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    elif phase == 'test':
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
    else:
        raise AssertionError('Not implemented yet')
    return transform


def get_cifar_dataset(data='cifar10', download_data=True):
    train_transform = make_transform_cifar(phase='train')
    test_transform = make_transform_cifar(phase='test')

    if data == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root='./cifar10', train=True, transform=train_transform,
                                                  download=download_data)
        test_data = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=test_transform,
                                                 download=download_data)
    elif data == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root='./cifar100', train=True, transform=train_transform,
                                                   download=download_data)
        test_data = torchvision.datasets.CIFAR100(root='./cifar100', train=False, transform=test_transform,
                                                  download=download_data)
    else:
        raise AssertionError('data has to be cifar10 or cifar100!')

    return train_data, test_data
