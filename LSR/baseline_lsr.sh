#!/usr/bin/env bash
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PYTHONPATH}:$PWD"
export CUDA_VISIBLE_DEVICES=0

export DATA_BASE='cub200'
export N_CLASSES=200
export NET='vgg16'
export LOGFILE="${DATA_BASE}.txt"

if [[ ! -d "model" ]]; then
    mkdir -p model
fi
if [[ ! -d "log" ]]; then
   mkdir -p log
fi

python LSR/baseline_lsr.py --dataset ${DATA_BASE} \
               --lr 1e-2 \
               --batch_size 64 \
               --weight_decay 1e-5 \
               --epochs 100 \
               --net ${NET} \
               --n_classes ${N_CLASSES} \
               --log ${LOGFILE} \
               --eps 0.1
