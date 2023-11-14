import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as vt

#import os
import yaml
from munch import DefaultMunch
from tqdm import trange
import random
import numpy as np
from wgan_gp import WGAN_GP


def set_seed(seed):
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


cfg_path = "config.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)

set_seed(cfg.seed)

transforms = vt.Compose([vt.ToTensor(),vt.Normalize(0.5, 0.5),
    vt.Resize((cfg.imsize, cfg.imsize),antialias=True)])

dataset = ImageFolder(
    root=cfg.data_dir, transform=transforms
)
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
model=WGAN_GP(cfg)

loop = trange(cfg.epochs, desc="Epoch: ", ncols=75)

for epoch in loop:
    model.train_epoch(dataloader)
    
    if (epoch+1) % cfg.save_model_freq == 0:
        model.save_model(epoch)
    if(epoch+1) % cfg.save_image_freq == 0:
        model.save_images(epoch,32)


