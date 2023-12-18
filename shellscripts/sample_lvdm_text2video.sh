#!/bin/bash

module load conda
conda activate /global/cfs/cdirs/m4244/stella/env_lvdm

#mkdir results/t2v
PROMPT="A welsh corgi is playing" # OR: PROMPT="input/prompts.txt" for sampling multiple prompts
OUTDIR="../results/object_centric"

#CKPT_PATH="../models/t2v/model.ckpt"
CKPT_PATH='/global/cfs/projectdirs/m4244/stella/object_LVDM/shellscripts/logs/231218_lvdm_short_sky_slot_consistency_scaled_temperature_1/checkpoints/epoch=0044-step=004004.ckpt'
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
    --object_centric True
