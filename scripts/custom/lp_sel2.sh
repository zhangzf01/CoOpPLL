#!/bin/bash

# custom config
DATA=/home/zzf/data

DATASET=$1
CFG=$2  # config file
LEVEL=$3
TRAINER=$4
GPU=$5

`echo "export CUDA_VISIBLE_DEVICES=${GPU}"`
export CUDA_VISIBLE_DEVICES=${GPU}

for SEED in 2
do
    for LR in 50 25 10 5 2.5 1 0.5 0.25 0.1 0.05 0.025 0.01
    do

        DIR=/home/zzf/doc/linearprobe_sel/${TRAINER}/${DATASET}/${CFG}/lr${LR}/partial_rate_${LEVEL}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train_sel_lr.py\
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/CLIP_LP_PLL/${CFG}.yaml \
            --output-dir ${DIR} \
            --level ${LEVEL}\
            --lr ${LR}\
            DATASET.NUM_SHOTS 16
        fi
    done
done