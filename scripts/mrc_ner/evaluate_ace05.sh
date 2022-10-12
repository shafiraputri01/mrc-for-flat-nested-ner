#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: eval.sh

REPO_PATH=/content/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/content/outputs/mrc_ner/ace2005/warmup0lr4e-5_drop0.3_norm1.0_weight0.1_warmup0_maxlen128
# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
BEST_CKPT_DEV=${OUTPUT_DIR}/epoch=14_v0.ckpt
PYTORCHLIGHT_HPARAMS=${OUTPUT_DIR}/lightning_logs/version_0/hparams.yaml
GPU_ID=0

python3 ${REPO_PATH}/evaluate/mrc_ner_evaluate.py ${BEST_CKPT_DEV} ${PYTORCHLIGHT_HPARAMS} ${GPU_ID}
