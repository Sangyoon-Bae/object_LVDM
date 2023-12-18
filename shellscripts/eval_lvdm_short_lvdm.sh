#!/bin/bash

#salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m4138_g


module load conda
conda activate /global/cfs/cdirs/m4244/stella/env_lvdm

DATACONFIG="/global/cfs/projectdirs/m4244/stella/object_LVDM/configs/lvdm_short/sky_lvdm.yaml"
FAKEPATH='/global/cfs/projectdirs/m4244/stella/object_LVDM/results/LVDM_whole/LVDM_whole.npy'
# need to be 2048x16x256x256x3
REALPATH='/global/cfs/projectdirs/m4244/stella/object_LVDM/datasets/sky_timelapse'
RESDIR='/global/cfs/projectdirs/m4244/stella/object_LVDM/results/fvd'

#mkdir -p $res_dir

#srun --ntasks-per-node=4 --nodes=1 --gpus-per-node=4
python /global/cfs/projectdirs/m4244/stella/object_LVDM/scripts/eval_cal_fvd_kvd.py \
    --yaml ${DATACONFIG} \
    --real_path ${REALPATH} \
    --fake_path ${FAKEPATH} \
    --batch_size 8 \
    --num_workers 4 \
    --n_runs 4 \
    --res_dir ${RESDIR} \
    --n_sample 8
