#!/bin/bash
set -ux

SAVE_DIR=outputs/DailyDialog_pointer2.infer
VOCAB_PATH=model/Bert/vocab.txt
DATA_DIR=data/DailyDialog
INIT_CHECKPOINT=outputs/DailyDialog/best.model
INIT_CHECKPOINT=outputs/DailyDialog_pointer2/best.model
DATA_TYPE=multi

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=2

# Paddle environment settings.
export FLAGS_fraction_of_gpu_memory_to_use=0.9
export FLAGS_eager_delete_scope=True
export FLAGS_eager_delete_tensor_gb=0.0
LD_LIBRARY_PATH=~/miniconda3/envs/plato/lib/
export LD_LIBRARY_PATH

python -u \
    ./preprocess.py \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE

python -u \
    ./run.py \
    --do_infer true \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE \
    --batch_size 1 \
    --num_type_embeddings 2 \
    --num_latent 20 \
    --use_discriminator true \
    --init_checkpoint $INIT_CHECKPOINT \
    --save_dir $SAVE_DIR \
    --use_pointer_network 2
