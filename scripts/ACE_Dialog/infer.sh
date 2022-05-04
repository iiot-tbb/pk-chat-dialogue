#!/bin/bash
set -ux

#SAVE_DIR=outputs/personchat_Dialog_ceshi.infer
#SAVE_DIR=outputs/ACE_Dialog_plato_3beam.infer
#SAVE_DIR=outputs/ACE_Dialog_gpt.infer
#SAVE_DIR=outputs/ACE_Dialog_gpt_du_correct.infer
#SAVE_DIR=outputs/PersonaChat_pointer2_context.infer
#SAVE_DIR=outputs/PersonaChat_Dialog_gpt.infer
#SAVE_DIR=outputs/DSTC7_pointer2_context.infer
#SAVE_DIR=outputs/DSTC_Dialog_gpt.infer
SAVE_DIR=outputs/DSTC7_plato.infer
#SAVE_DIR=outputs/DailyDialog_gpt.infer
#SAVE_DIR=outputs/DailyDialog_pointer2_context.infer
VOCAB_PATH=model/Bert/vocab.txt
DATA_DIR=data/DSTC7_AVSD
#DATA_DIR=data/DailyDialog
#DATA_DIR=data/PersonaChat
#DATA_DIR=data/ACE_Dialog/data_du
#INIT_CHECKPOINT=outputs/personchat_Dialog_gpt/best.model
#INIT_CHECKPOINT=outputs/ACE_Dialog_gpt/best.model
#INIT_CHECKPOINT=outputs/PersonaChat_pointer2_context/best.model
#INIT_CHECKPOINT=outputs/DSTC7_pointer2_context/best.model
INIT_CHECKPOINT=outputs/DSTC7_plato/best.model
#INIT_CHECKPOINT=outputs/DSTC_Dialog_gpt/best.model
#INIT_CHECKPOINT=outputs/ACE_Dialog_gpt/best.model
#INIT_CHECKPOINT=outputs/DailyDialog_gpt/best.model
#INIT_CHECKPOINT=outputs/DailyDialog_pointer2_context/best.model
DATA_TYPE=multi_knowledge
#DATA_TYPE=multi

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=1
LD_LIBRARY_PATH=~/miniconda3/envs/plato/lib/
export LD_LIBRARY_PATH

# Paddle environment settings.
export FLAGS_fraction_of_gpu_memory_to_use=0.9
export FLAGS_eager_delete_scope=True
export FLAGS_eager_delete_tensor_gb=0.0

# python -u \
#     ./preprocess.py \ #前处理文件，把原始的对话文件转换成对应的encoding文件
#     --vocab_path $VOCAB_PATH \
#     --data_dir $DATA_DIR \
#     --data_type $DATA_TYPE

if [[ ! -e $DATA_DIR/dial.train.jsonl ]]; then
    python -u \
        ./preprocess.py \
        --vocab_path $VOCAB_PATH \
        --data_dir $DATA_DIR \
        --data_type $DATA_TYPE
fi

python -u \
    ./run.py \
    --do_infer true \
    --vocab_path $VOCAB_PATH \
    --data_dir $DATA_DIR \
    --data_type $DATA_TYPE \
    --batch_size 1 \
    --num_type_embeddings 3 \
    --use_discriminator false \
    --init_checkpoint $INIT_CHECKPOINT \
    --save_dir $SAVE_DIR \
    --weight_sharing true \
    --bidirectional_context true \
    --use_pointer_network -1 \
    --beam_size 3
    #--init_checkpoint $INIT_CHECKPOINT \
    #--generator GreedySampling \
    
