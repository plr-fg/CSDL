# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from net.network import make_network
from utils.eval import evaluate
from utils.utils import make_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

torch.manual_seed(0)
torch.cuda.manual_seed(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_classes', type=int, required=True)
    parser.add_argument('--net', type=str, required=True)
    args = parser.parse_args()

    Net, _ = make_network(args.net)
    net = Net(n_classes=args.n_classes, pretrained=False)
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(args.model))

    test_transform = make_transform(phase='test', output_size=448)
    test_data = torchvision.datasets.ImageFolder(os.path.join(args.data, 'val'), transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    test_accuracy = evaluate(test_loader, net)
    print('===>>> Test accuracy: {}'.format(test_accuracy))
