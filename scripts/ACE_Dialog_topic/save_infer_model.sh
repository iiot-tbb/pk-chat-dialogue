#!/bin/bash
set -ux

#SAVE_DIR=outputs/personchat_Dialog_ceshi.infer
SAVE_DIR=outputs/ACE_Dialog_context_transfer.infer
#SAVE_DIR=outputs/ACE_Dialog_gpt.infer
VOCAB_PATH=model/Bert/vocab.txt
#DATA_DIR=data/DSTC7_AVSD
#DATA_DIR=data/PersonaChat
DATA_DIR=data/ACE_Dialog_topic
#INIT_CHECKPOINT=outputs/personchat_Dialog_gpt/best.model
#INIT_CHECKPOINT=outputs/ACE_Dialog_pointer_context_transfer2/best.model
#INIT_CHECKPOINT=outputs/ACE_Dialog_gpt/best.model
DATA_TYPE=multi_knowledge_topic_transfer

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=3
LD_LIBRARY_PATH=~/miniconda3/envs/plato/lib/
export LD_LIBRARY_PATH

# Paddle environment settings.
export FLAGS_fraction_of_gpu_memory_to_use=0.9
export FLAGS_eager_delete_scope=True
export FLAGS_eager_delete_tensor_gb=0.0

python -u \
    ./preprocess.py \ #前处理文件，把原始的对话文件转换成对应的encoding文件
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE

python -u \
    ./save_model.py \
    --do_infer true \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE \
    --batch_size 1 \
    --num_type_embeddings 3 \
    --use_discriminator false \
    --save_dir $SAVE_DIR \
    --weight_sharing true \
    --bidirectional_context true \
    --use_pointer_network 2 \
    --use_topic_trans_judge true \
    --use_topic_evaluate true
    #--init_checkpoint $INIT_CHECKPOINT \
    #--generator GreedySampling \
    #--beam_size 1 \
    #--init_checkpoint $INIT_CHECKPOINT \
