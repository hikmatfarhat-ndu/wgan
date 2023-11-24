import torch
from networks import Generator
from utils import random_sample
import yaml
from munch import DefaultMunch
import os
from PIL import Image
import numpy as np
from torchvision.utils import make_grid
import torch.functional as F
cfg_path = "config.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)
generator=Generator(
            z_dim=cfg.z_dim,
            out_ch=cfg.img_ch,norm_type=cfg.g_norm_type,
            final_activation=cfg.g_final_activation
        )
dir_list=os.listdir(cfg.weights_dir)
if dir_list:
            gen_files=[f for f in dir_list if f.startswith("gen")]
            dis_files=[f for f in dir_list if f.startswith("dis")]
            gen_files.sort()
            dis_files.sort()

            generator.load_state_dict(torch.load(cfg.weights_dir+"/"+gen_files[-1]))
           
            print(f"loaded weights from {cfg.weights_dir}/{gen_files[-1]} and {cfg.weights_dir}/{dis_files[-1]}")
else:
    exit("No weights found")

def recover_image(tensor):
        # PIL expects the image to be of shape (H,W,C)
        # in PyTorch it's (C,H,W)
        # min=tensor.min()
        # max=tensor.max()
        # tensor=tensor-min
        # tensor=tensor/max
        tensor=tensor.cpu().numpy().transpose(1, 2,0)*255
        
        return Image.fromarray(tensor.astype(np.uint8))
    
    
generator.to(cfg.device)
generator.eval()
noise = random_sample(cfg.batch_size,cfg.z_dim,cfg.device)

fake_images = generator(noise)
with torch.no_grad():
    fake_images = generator(noise)
    for i in range(fake_images.shape[0]):
        img=make_grid(fake_images[i],nrow=1,normalize=True)
        img=recover_image(img)
        img.save(os.path.join(cfg.samples_dir,f"sample_{i}.png"))

#print(fake_images.shape)