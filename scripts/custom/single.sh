#!/bin/bash


# custom config
DATA=/home/zzf/data

DATASET=$1
CFG=$2  # backbone
NCTX=$3  # number of context tokens
LEVEL=$4
TASK=$5
TRAINER=$6
GPU=$7

`echo "export CUDA_VISIBLE_DEVICES=${GPU}"`
export CUDA_VISIBLE_DEVICES=${GPU}

for SEED in 4
do
    DIR=/home/zzf/doc/${TASK}/${TRAINER}/${DATASET}/${CFG}_16shots_nctx${NCTX}/partial_rate_${LEVEL}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py\
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/CoOp_PLL/${CFG}.yaml \
        --output-dir ${DIR} \
        --level ${LEVEL}\
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC False \
        TRAINER.COOP.CLASS_TOKEN_POSITION end \
        DATASET.NUM_SHOTS 16
    fi
done