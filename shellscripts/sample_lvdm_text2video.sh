#!/bin/bash

#mkdir results/t2v
PROMPT="A welsh corgi is playing" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="../results/object_centric"

CKPT_PATH="../models/t2v/model.ckpt"
CONFIG_PATH="../configs/lvdm_short/text2video.yaml"

python ../scripts/sample_text2video.py \
    --ckpt_path $CKPT_PATH \
    --config_path $CONFIG_PATH \
    --prompt "$PROMPT" \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 1000 \
    --show_denoising_progress \
    --save_jpg \
    --object_centric
