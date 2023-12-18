#!/bin/bash

module load conda
conda activate /global/cfs/cdirs/m4244/stella/env_lvdm

srun --ntasks-per-node=4 --nodes=1 --gpus-per-node=4 python /global/cfs/projectdirs/m4244/stella/object_LVDM/main.py \
--base "/global/cfs/projectdirs/m4244/stella/object_LVDM/configs/lvdm_short/sky.yaml" \
-t --gpus 0,1,2,3 --nnodes=1 \
--name "231218_lvdm_short_sky_slot_consistency_scaled_temperature_1" \
--auto_resume True \
--load_from_checkpoint '/global/cfs/projectdirs/m4244/stella/object_LVDM/shellscripts/logs/231218_lvdm_short_sky_slot_consistency_scaled_temperature_1/checkpoints/epoch=0044-step=004004.ckpt'