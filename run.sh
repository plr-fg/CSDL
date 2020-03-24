#!/usr/bin/env bash
export PYTHONWARNINGS="ignore"
export PYTHONPATH="${PYTHONPATH}:$PWD"
export CUDA_VISIBLE_DEVICES=0

export DATA_BASE='cub200'
export N_CLASSES=200
export NET='vgg16'

# --- config
export INIT_W=0.5
export END_W=0.1
export EPOCH_GRADUAL_W=10
export EPS=0.1
export LAMDA=0.003
export SIMILARITY='cosine'
# --- config

export LOGFILE="${DATA_BASE}-${SIMILARITY}_similarity-InitW${INIT_W}_EndW${END_W}_T${EPOCH_GRADUAL_W}-eps${EPS}-lambda${LAMDA}.txt"

if [[ ! -d "model" ]]; then
    mkdir -p model
fi
if [[ ! -d "log" ]]; then
   mkdir -p log
fi

python main.py --dataset ${DATA_BASE} \
               --lr 1e-2 \
               --batch_size 64 \
               --weight_decay 1e-4 \
               --epochs 100 \
               --net ${NET} \
               --n_classes ${N_CLASSES} \
               --log ${LOGFILE} \
               --init_w ${INIT_W} \
               --end_w ${END_W} \
               --epoch_gradual_w ${EPOCH_GRADUAL_W} \
               --eps ${EPS} \
               --lamda ${LAMDA} \
               --similarity ${SIMILARITY}
