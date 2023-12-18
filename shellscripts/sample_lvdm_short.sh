
#!/bin/bash

module load conda
conda activate /global/cfs/cdirs/m4244/stella/env_lvdm

CONFIG_PATH="/global/cfs/projectdirs/m4244/stella/object_LVDM/configs/lvdm_short/sky_lvdm.yaml"
BASE_PATH="/global/cfs/projectdirs/m4244/stella/object_LVDM/models/lvdm_short/short_sky.ckpt"
AEPATH="/global/cfs/projectdirs/m4244/stella/object_LVDM/models/ae/ae_sky.ckpt"
OUTDIR="/global/cfs/projectdirs/m4244/stella/object_LVDM/results/uncond_short/"

mkdir -p $OUTDIR


python /global/cfs/projectdirs/m4244/stella/object_LVDM/scripts/sample_uncond.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 2500 \
    --show_denoising_progress \
    --object_centric False \
    model.params.first_stage_config.params.ckpt_path=$AEPATH

python /global/cfs/projectdirs/m4244/stella/object_LVDM/scripts/sample_uncond.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 5500 \
    --show_denoising_progress \
    --object_centric False \
    model.params.first_stage_config.params.ckpt_path=$AEPATH

python /global/cfs/projectdirs/m4244/stella/object_LVDM/scripts/sample_uncond.py \
    --ckpt_path $BASE_PATH \
    --config_path $CONFIG_PATH \
    --save_dir $OUTDIR \
    --n_samples 1 \
    --batch_size 1 \
    --seed 6500 \
    --show_denoising_progress \
    --object_centric False \
    model.params.first_stage_config.params.ckpt_path=$AEPATH

# if use DDIM： add: `--sample_type ddim --ddim_steps 50`
