#!/bin/bash
# mkdir -p models/ae
# mkdir -p models/lvdm_short
# mkdir -p models/t2v

# # sky timelapse
# wget -O models/ae/ae_sky.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/ae/ae_sky.ckpt
wget -O models/lvdm_short/short_sky.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/lvdm_short/short_sky.ckpt  

# taichi
wget -O models/ae/ae_taichi.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/ae/ae_taichi.ckpt
wget -O models/lvdm_short/short_taichi.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/lvdm_short/short_taichi.ckpt

# text2video
wget -O models/t2v/model.ckpt https://huggingface.co/Yingqing/LVDM/resolve/main/lvdm_short/t2v.ckpt