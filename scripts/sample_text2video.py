import os
import time
import argparse
import yaml, math
from tqdm import trange
import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
import torch.distributed as dist
from pytorch_lightning import seed_everything

import sys
sys.path.append('..')

from lvdm.samplers.ddim import DDIMSampler
from lvdm.utils.common_utils import str2bool
from lvdm.utils.dist_utils import setup_dist, gather_data
from scripts.sample_utils import (load_model, 
                                  get_conditions, make_model_input_shape, torch_to_np, sample_batch, 
                                  save_results,
                                  save_args
                                  )
from lvdm.models.modules.slot_attention import SlotAttention


# ------------------------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()
    # basic args
    parser.add_argument("--ckpt_path", type=str, help="model checkpoint path")
    parser.add_argument("--config_path", type=str, help="model config path (a yaml file)")
    parser.add_argument("--prompt", type=str, help="input text prompts for text2video (a sentence OR a txt file).")
    parser.add_argument("--save_dir", type=str, help="results saving dir", default="results/")
    # device args
    parser.add_argument("--ddp", action='store_true', help="whether use pytorch ddp mode for parallel sampling (recommend for multi-gpu case)", default=False)
    parser.add_argument("--local_rank", type=int, help="is used for pytorch ddp mode", default=0)
    parser.add_argument("--gpu_id", type=int, help="choose a specific gpu", default=0)
    # sampling args
    parser.add_argument("--n_samples", type=int, help="how many samples for each text prompt", default=2)
    parser.add_argument("--batch_size", type=int, help="video batch size for sampling", default=1)
    parser.add_argument("--decode_frame_bs", type=int, help="frame batch size for framewise decoding", default=1)
    parser.add_argument("--sample_type", type=str, help="ddpm or ddim", default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, help="ddim sampling -- number of ddim denoising timesteps", default=50)
    parser.add_argument("--eta", type=float, help="ddim sampling -- eta (0.0 yields deterministic sampling, 1.0 yields random sampling)", default=1.0)
    parser.add_argument("--seed", type=int, default=None, help="fix a seed for randomness (If you want to reproduce the sample results)")
    parser.add_argument("--num_frames", type=int, default=16, help="number of input frames")
    parser.add_argument("--show_denoising_progress", action='store_true', default=False, help="whether show denoising progress during sampling one batch",)
    parser.add_argument("--cfg_scale", type=float, default=15.0, help="classifier-free guidance scale")
    # saving args
    parser.add_argument("--save_mp4", type=str2bool, default=True, help="whether save samples in separate mp4 files", choices=["True", "true", "False", "false"])
    parser.add_argument("--save_mp4_sheet", action='store_true', default=False, help="whether save samples in mp4 file",)
    parser.add_argument("--save_npz", action='store_true', default=False, help="whether save samples in npz file",)
    parser.add_argument("--save_jpg", action='store_true', default=False, help="whether save samples in jpg file",)
    parser.add_argument("--save_fps", type=int, default=8, help="fps of saved mp4 videos",)
    parser.add_argument('--object_centric', type=str2bool, default=True)
    return parser

# ------------------------------------------------------------------------------------------
def slot_loss (slot_list, i, j, length, temperature):
    # 가까운 frame끼리 similarity가 크다! 따라서, 두 frame이 가까울수록 penalty 곱하기 전 -log(분수) 값이 작아야 함.
    # penalty term은 가까운 frame을 더 가깝게 (-log(분수) 값이 작게) 하는 역할. i, j가 가까울수록 penalty term이 작아야 함.
    sim = nn.CosineSimilarity(dim=0, eps=1e-6)
    numerator = torch.exp(sim(slot_list[i], slot_list[j])/temperature)
    complent_list = list(range(length))[:j]+list(range(length))[j+1:]
    denominator = sum([torch.exp(sim(slot_list[i], slot_list[k])/temperature) for k in complent_list])
    penalty = np.abs(i-j)
    
    return -torch.log((numerator/denominator))*penalty


@torch.no_grad()
def sample_text2video(model, prompt, n_samples, batch_size,
                      sample_type="ddim", sampler=None, 
                      ddim_steps=50, eta=1.0, cfg_scale=7.5, 
                      decode_frame_bs=1,
                      ddp=False, all_gather=True, 
                      batch_progress=True, show_denoising_progress=False,
                      num_frames=None,
                      ):
    # get cond vector
    assert(model.cond_stage_model is not None)
    cond_embd = get_conditions(prompt, model, batch_size)
    uncond_embd = get_conditions("", model, batch_size) if cfg_scale != 1.0 else None
    args = get_parser().parse_args()
    if args.object_centric:
        slot_attn = SlotAttention(
                                  num_slots = 1,
                                  dim = 4,
                                  iters = 3   # iterations of attention, defaults to 3
                                  )
        # input is (batch, embedding ssize, dim)
    
    # sample batches
    all_videos = []
    n_iter = math.ceil(n_samples / batch_size)
    iterator  = trange(n_iter, desc="Sampling Batches (text-to-video)") if batch_progress else range(n_iter)
    for _ in iterator:
        noise_shape = make_model_input_shape(model, batch_size, T=num_frames)
        samples_latent = sample_batch(model, noise_shape, cond_embd,
                                            sample_type=sample_type,
                                            sampler=sampler,
                                            ddim_steps=ddim_steps,
                                            eta=eta,
                                            unconditional_guidance_scale=cfg_scale, 
                                            uc=uncond_embd,
                                            denoising_progress=show_denoising_progress,
                                            )
        # 여기서 slot을 수행하는 것도 방법..! sampler에서 수행해서 나중에 train까지 옮기자구.
        if args.object_centric:
            device = samples_latent.get_device()
            slot_attn.to(device)
            ## shape of samples_latent is:torch.Size([1, 4, 16, 32, 32]) 4는 channel, 16은 length, 32는 image size.
            length = samples_latent.shape[2]
            channel = samples_latent.shape[1]
            image_size = samples_latent.shape[-1]
            slot_list = []
            for i in range(length):
                #slot = slot_attn(samples_latent[:, :, i, :, :].reshape(1, channel, image_size**2)) # (1, 1, 1024)
                slot = slot_attn(samples_latent[:, :, i, :, :].permute(0,2,3,1).reshape(1, image_size**2, channel)) # (1, 1024, 1)
                slot_list.append(slot.flatten()) # 1024
                
            for i in range(length):
                for j in range(length):
                    if i<j:
                        print('slot loss between frame {0} and {1} is:'.format(str(i), str(j)), slot_loss(slot_list, i, j, length, 5))
        

        samples = model.decode_first_stage(samples_latent, decode_bs=decode_frame_bs, return_cpu=False)
        # samples가 video임
        
        # gather samples from multiple gpus
        if ddp and all_gather:
            data_list = gather_data(samples, return_np=False)
            all_videos.extend([torch_to_np(data) for data in data_list])
        else:
            all_videos.append(torch_to_np(samples))
    
    all_videos = np.concatenate(all_videos, axis=0)
    assert(all_videos.shape[0] >= n_samples)
    return all_videos



# ------------------------------------------------------------------------------------------
def main():
    """
    text-to-video generation
    """
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    os.makedirs(opt.save_dir, exist_ok=True)
    save_args(opt.save_dir, opt)
    
    # set device
    if opt.ddp:
        setup_dist(opt.local_rank)
        opt.n_samples = math.ceil(opt.n_samples / dist.get_world_size())
        gpu_id = None
    else:
        gpu_id = opt.gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
    
    # set random seed
    if opt.seed is not None:
        if opt.ddp:
            seed = opt.local_rank + opt.seed
        else:
            seed = opt.seed
        seed_everything(seed)

    # load & merge config
    config = OmegaConf.load(opt.config_path)
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(config, cli)
    print("config: \n", config)

    # get model & sampler
    model, _, _ = load_model(config, opt.ckpt_path)
    ddim_sampler = DDIMSampler(model) if opt.sample_type == "ddim" else None
    
    # prepare prompt
    if opt.prompt.endswith(".txt"):
        opt.prompt_file = opt.prompt
        opt.prompt = None
    else:
        opt.prompt_file = None

    if opt.prompt_file is not None:
        f = open(opt.prompt_file, 'r')
        prompts, line_idx = [], []
        for idx, line in enumerate(f.readlines()):
            l = line.strip()
            if len(l) != 0:
                prompts.append(l)
                line_idx.append(idx)
        f.close()
        cmd = f"cp {opt.prompt_file} {opt.save_dir}"
        os.system(cmd)
    else:
        prompts = [opt.prompt]
        line_idx = [None]

    # go
    start = time.time()  
    for prompt in prompts:
        # sample
        samples = sample_text2video(model, prompt, opt.n_samples, opt.batch_size,
                          sample_type=opt.sample_type, sampler=ddim_sampler,
                          ddim_steps=opt.ddim_steps, eta=opt.eta, 
                          cfg_scale=opt.cfg_scale,
                          decode_frame_bs=opt.decode_frame_bs,
                          ddp=opt.ddp, show_denoising_progress=opt.show_denoising_progress,
                          num_frames=opt.num_frames,
                          )
        # save
        if (opt.ddp and dist.get_rank() == 0) or (not opt.ddp):
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            save_name = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            if opt.seed is not None:
                if opt.object_centric:
                    save_name = save_name + f"OLVDM_seed{seed:05d}"
                else:
                    save_name = save_name + f"LVDM_seed{seed:05d}"
            save_results(samples, opt.save_dir, save_name=save_name, save_fps=opt.save_fps)
    print("Finish sampling!")
    print(f"Run time = {(time.time() - start):.2f} seconds")

    if opt.ddp:
        dist.destroy_process_group()


# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()