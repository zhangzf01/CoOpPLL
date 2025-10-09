#!/bin/bash

# nohup bash scripts/lin_probe/main.sh caltech101 rn50 16 3 & 

# custom config
DATA=/home/zzf/data
TRAINER=LinearProbe_v1

DATASET=$1
CFG=$2  # config file
SHOTS=$3  # number of shots (1, 2, 4, 8, 16)
LEVEL=$4
LR=$5
GPU=$6
`echo "export CUDA_VISIBLE_DEVICES=${GPU}"`
export CUDA_VISIBLE_DEVICES=${GPU}

for SEED in 1
do
    DIR=/home/zzf/doc/foudation_pll/output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/lr${LR}/pll_${LEVEL}/seed${SEED}
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
        DATASET.NUM_SHOTS ${SHOTS}

    fi
done