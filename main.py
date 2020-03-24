# -*- coding: utf-8 -*-

import time
import argparse
from train import Trainer


def show_params(params):
    data_path = params['data_base']
    print('|-----------------------------------------------------')
    print('| Training Config : ')
    print('| net             : {}'.format(params['net']))
    print('| learning_rate   : {}'.format(params['lr']), end=', ')
    print('decays with ratio {} in every {} epochs'.format(params['lr_gamma'], params['lr_step']))
    print('| weight_decay    : {}'.format(params['weight_decay']))
    print('| batch_size      : {}'.format(params['batch_size']))
    print('| epochs          : {}'.format(params['epochs']))
    print('| num of classes  : {}'.format(params['n_classes']))
    print('|-----------------------------------------------------')
    print('| init_w          : {}'.format(params['init_w']))
    print('| end_w           : {}'.format(params['end_w']))
    print('| epoch_gradual_w : {}'.format(params['epoch_gradual_w']))
    print('| epsilon         : {}'.format(params['eps']))
    print('| lamda           : {}'.format(params['lamda']))
    print('| similarity      : {}'.format(params['similarity']))
    print('|-----------------------------------------------------')
    print('| data_path  : {}'.format(data_path))
    print('| log_file   : {}'.format(params['log']))
    print('|-----------------------------------------------------')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_step', type=int, default=10)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--net', type=str, default='bcnn')
    parser.add_argument('--n_classes', type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--log', type=str, default='logfile.txt')
    parser.add_argument('--print_freq', type=int, default=30)

    parser.add_argument('--init_w', type=float, default=0.1)
    parser.add_argument('--end_w', type=float, default=0.1)
    parser.add_argument('--epoch_gradual_w', type=int, default=20)
    parser.add_argument('--eps', type=float, default=0.0)
    parser.add_argument('--lamda', type=float, default=0.003)
    parser.add_argument('--similarity', type=str, default='cosine')

    args = parser.parse_args()
    logfile = 'log/{}'.format(args.log)

    config = {
        'data_base': args.dataset,
        'lr': args.lr,
        'lr_step': args.lr_step,
        'lr_gamma': args.lr_gamma,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'net': args.net,
        'n_classes': args.n_classes,
        'resume': args.resume,
        'log': logfile,
        'print_freq': args.print_freq,

        'init_w': args.init_w,
        'end_w': args.end_w,
        'epoch_gradual_w': args.epoch_gradual_w,
        'eps': args.eps,
        'lamda': args.lamda,
        'similarity': args.similarity,
    }

    show_params(config)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end-start))
