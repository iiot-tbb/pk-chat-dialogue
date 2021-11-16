#!/bin/bash
set -ux

SAVE_DIR=outputs/ACE_Dialog_gpt
VOCAB_PATH=model/Bert/vocab.txt
DATA_DIR=data/ACE_Dialog
INIT_CHECKPOINT=model/PLATO
#INIT_CHECKPOINT=outputs/ACE_Dialog/best.model
DATA_TYPE=multi_knowledge


# Paddle environment settings.

python -u \
    ./preprocess.py \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE