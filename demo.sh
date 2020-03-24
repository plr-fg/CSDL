#!/usr/bin/env bash
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PYTHONPATH}:$PWD"
export CUDA_VISIBLE_DEVICES=0

export DATA_BASE='data'
export MODEL='model/cub200-vgg16-best_epoch-82.31.pth'
export N_CLASSES=200
export NET='vgg16'

python demo.py --data ${DATA_BASE} --model ${MODEL} --n_classes ${N_CLASSES} --net ${NET}
