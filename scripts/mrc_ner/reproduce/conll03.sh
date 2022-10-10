#!/usr/bin/env bash
# -*- coding: utf-8 -*-

TIME=0901
FILE=conll03_cased_large
REPO_PATH=/content/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/content/dataset/conll03
BERT_DIR=/content/bert/bert_cased_large
OUTPUT_BASE=/content/outputs

BATCH=8
GRAD_ACC=2
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=4e-5
LR_MINI=5e-7
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=128
MAX_NORM=1.0
MAX_EPOCH=15
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=torch.adam
VAL_CHECK=0.25
PREC=16
SPAN_CAND=pred_and_gold


OUTPUT_DIR=${OUTPUT_BASE}/mrc_ner/${TIME}/${FILE}_cased_large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=0 python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="1" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--distributed_backend=ddp \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--flat \
--lr_mini ${LR_MINI}

