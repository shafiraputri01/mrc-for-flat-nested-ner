#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: flat_inference_conll03.sh
#

REPO_PATH=/content/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_SIGN=conll03
DATA_DIR=/content/dataset/conll03
BERT_DIR=/content/bert/bert_cased_large
MAX_LEN=180
MODEL_CKPT=/content/drive/MyDrive/outputs/outputs/mrc_ner/0901/conll03_cased_large_cased_large_lr4e-5_drop0.3_norm_weight0.1_warmup0_maxlen/epoch=14.ckpt
HPARAMS_FILE=/content/drive/MyDrive/outputs/outputs/mrc_ner/0901/conll03_cased_large_cased_large_lr4e-5_drop0.3_norm_weight0.1_warmup0_maxlen/lightning_logs/version_0/hparams.yaml


python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--flat_ner \
--dataset_sign ${DATA_SIGN}
