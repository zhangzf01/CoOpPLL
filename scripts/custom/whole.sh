#!/bin/bash


# custom config
DATA=/home/zzf/data


DATASET=$1
CFG=$2  # backbone
NCTX=$3  # number of context tokens
TASK=$4
TRAINER=$5
GPU=$6

`echo "export CUDA_VISIBLE_DEVICES=${GPU}"`
export CUDA_VISIBLE_DEVICES=${GPU}


for SEED in 1 2 3 4
do
    for PARTIAL_RATE in 0.3 0.5 0.7
    do
        DIR=/home/zzf/doc/${TASK}/${TRAINER}/${DATASET}/${CFG}_16shots_nctx${NCTX}/partial_rate_${PARTIAL_RATE}/seed${SEED}
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
            --level ${PARTIAL_RATE}\
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.CSC False \
            TRAINER.COOP.CLASS_TOKEN_POSITION end \
            DATASET.NUM_SHOTS 16
        fi
    done
done