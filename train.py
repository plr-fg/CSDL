# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.datasets import ImageFolder
import numpy as np
import random
from PIL import ImageFile
from loss import PseudoLabelCrossEntropyLoss, get_init_center
from net.network import make_network
from utils.utils import print_to_console, print_to_logfile, make_transform, make_optimizer, get_lr_from_optimizer
from utils.metrics import AverageMeter, accuracy
from utils.eval import evaluate
from utils.cifar import get_cifar_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


class Trainer(object):
    def __init__(self, config):
        # Config
        self._config = config
        self._epochs = config['epochs']
        self._logfile = config['log']
        self._n_classes = config['n_classes']

        # Network
        Net, feature_dim = make_network(config['net'])
        self._n_feature_dim = feature_dim
        net = Net(n_classes=self._n_classes, pretrained=True)
        # Move network to cuda
        print('| Number of available GPUs : {} ({})'.format(torch.cuda.device_count(),
                                                            os.environ["CUDA_VISIBLE_DEVICES"]))
        if torch.cuda.device_count() >= 1:
            self._net = nn.DataParallel(net).cuda()
        else:
            raise AssertionError('CPU version is not implemented yet!')

        # Loss Criterion
        self._criterion = PseudoLabelCrossEntropyLoss(n_classes=self._n_classes, feature_dim=self._n_feature_dim,
                                                      epsilon=config['eps'], lamda=config['lamda'],
                                                      similarity=config['similarity']).cuda()

        self.w_pseudo_label_plan = [config['end_w']] * config['epochs']
        self.w_pseudo_label_plan[:config['epoch_gradual_w']] = list(np.linspace(config['init_w'],
                                                                                config['end_w'],
                                                                                config['epoch_gradual_w']))

        self.center_param_optimizer = optim.SGD(self._criterion.parameters(), lr=0.5)

        # Optimizer
        params_to_optimize = self._net.parameters()
        self._optimizer = make_optimizer(params_to_optimize, lr=config['lr'], weight_decay=config['weight_decay'],
                                         opt='SGD')

        self._scheduler = optim.lr_scheduler.StepLR(self._optimizer,
                                                    step_size=config['lr_step'], gamma=config['lr_gamma'])

        # metrics
        self._train_loss = AverageMeter()
        self._train_accuracy = AverageMeter()

        # Dataloader
        if config['data_base'] == 'cifar10' or config['data_base'] == 'cifar100':
            train_data, test_data = get_cifar_dataset(data=config['data_base'], download_data=True)
        else:
            train_transform = make_transform(phase='train', output_size=448)
            test_transform = make_transform(phase='test', output_size=448)
            train_data = ImageFolder(os.path.join(config['data_base'], 'train'), transform=train_transform)
            test_data = ImageFolder(os.path.join(config['data_base'], 'val'), transform=test_transform)
        self._train_loader = data.DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=4,
                                             pin_memory=True)
        self._test_loader = data.DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
        print('|-----------------------------------------------------')
        print('| Number of samples in train set : {}'.format(len(train_data)))
        print('| Number of samples in test  set : {}'.format(len(test_data)))
        print('| Number of classes in train set : {}'.format(len(train_data.classes)))
        print('| Number of classes in test  set : {}'.format(len(test_data.classes)))
        print('|-----------------------------------------------------')
        assert len(train_data.classes) == self._n_classes and \
            len(test_data.classes) == self._n_classes, 'number of classes is wrong'

        # Resume or not
        if config['resume']:
            assert os.path.isfile('checkpoint.pth'), 'no checkpoint.pth exists!'
            print('---> loading checkpoint.pth <---')
            checkpoint = torch.load('checkpoint.pth')
            self._start_epoch = checkpoint['epoch']
            self._best_accuracy = checkpoint['best_accuracy']
            self._best_epoch = checkpoint['best_epoch']
            self._net.load_state_dict(checkpoint['state_dict'])
            self._optimizer.load_state_dict(checkpoint['optimizer'])
            self._scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            print('---> no checkpoint loaded <---')
            self._start_epoch = 0
            self._best_accuracy = 0.0
            self._best_epoch = None

    def train(self):
        print('---> calculating the init of centers! <---')
        mean_feature = get_init_center(self._net, self._train_loader, self._n_classes,
                                       n_dim_feature=self._n_feature_dim)
        self._criterion.init_centers(mean_feature.cuda())

        console_header = 'Epoch\tTrain_Loss\tTrain_Accuracy\tTest_Accuracy\tEpoch_Runtime\tLearning_Rate'
        print_to_console(console_header)
        print_to_logfile(self._logfile, console_header, init=True)

        for t in range(self._start_epoch, self._epochs):
            epoch_start = time.time()
            # reset average meters
            self._train_loss.reset()
            self._train_accuracy.reset()

            self._net.train(True)
            self.single_epoch_training(t, log_iter=True, log_freq=self._config['print_freq'])
            test_accuracy = evaluate(self._test_loader, self._net)

            lr = get_lr_from_optimizer(self._optimizer)

            if test_accuracy > self._best_accuracy:
                self._best_accuracy = test_accuracy
                self._best_epoch = t + 1
                torch.save(self._net.state_dict(), 'model/best_epoch.pth')
                # print('*', end='')
            epoch_end = time.time()
            single_epoch_runtime = epoch_end - epoch_start
            # Logging
            console_content = '{:05d}\t{:10.4f}\t{:14.4f}\t{:13.4f}\t{:13.2f}\t{:13.1e}'.format(
                t + 1, self._train_loss.avg, self._train_accuracy.avg, test_accuracy, single_epoch_runtime, lr)
            print_to_console(console_content)
            print_to_logfile(self._logfile, console_content, init=False)

            self._scheduler.step()
            # save checkpoint
            save_checkpoint({
                'epoch': t + 1,
                'state_dict': self._net.state_dict(),
                'best_epoch': self._best_epoch,
                'best_accuracy': self._best_accuracy,
                'optimizer': self._optimizer.state_dict(),
                'scheduler': self._scheduler.state_dict()
            })

        console_content = 'Best at epoch {}, test accuracy is {}'.format(self._best_epoch, self._best_accuracy)
        print_to_console(console_content)

        # rename log file
        os.rename(self._logfile, self._logfile.replace('.txt', '-{}_{}_{}_{:.4f}.txt'.format(
            self._config['net'], self._config['batch_size'], self._config['lr'], self._best_accuracy)))
        # rename model file
        model_path = 'model/best_epoch.pth'
        new_model_path = 'model/{}-{}-best_epoch-{:.2f}.pth'.format(self._config['data_base'],
                                                                    self._config['net'], self._best_accuracy)
        if os.path.isfile(model_path):
            os.rename(model_path, new_model_path)

    def single_epoch_training(self, epoch, log_iter=True, log_freq=1):
        s = time.time()
        for it, (x, y) in enumerate(self._train_loader):
            # s = time.time()

            x = x.cuda()
            y = y.cuda()
            self._optimizer.zero_grad()
            self.center_param_optimizer.zero_grad()
            embeddings, logits = self._net(x)
            loss = self._criterion(embeddings, logits, y, w=self.w_pseudo_label_plan[epoch])

            train_accuracy = accuracy(logits, y, topk=(1,))

            self._train_loss.update(loss.item(), x.size(0))
            self._train_accuracy.update(train_accuracy[0], x.size(0))

            loss.backward()
            self._optimizer.step()
            for param in self._criterion.parameters():
                param.grad.data *= 0.5
            self.center_param_optimizer.step()

            if log_iter and (it+1) % log_freq == 0:
                e = time.time()
                console_content = 'Epoch:[{0:03d}/{1:03d}]  Iter:[{2:04d}/{3:04d}]  ' \
                                  'Train Accuracy :[{4:6.2f}]  Train Loss:[{5:4.4f}]  ' \
                                  '({6:3.2f} sec/iter)'.format(epoch + 1, self._epochs, it + 1,
                                                               len(self._train_loader),
                                                               self._train_accuracy.avg,
                                                               self._train_loss.avg, (e - s)/log_freq)
                print_to_console(console_content)
                s = time.time()
