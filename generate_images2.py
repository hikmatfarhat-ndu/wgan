import torch
from networks import Generator
from utils import random_sample
import yaml
from munch import DefaultMunch
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as vt
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from PIL import Image
from torchvision.utils import make_grid 
import numpy as np
from matplotlib import pyplot as plt
from torcheval import metrics


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--weights-path", type=str, help="weights path")
args=parser.parse_args()

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

if args.weights_path is None:

    if dir_list:# PIL expects the image to be of shape (H,W,C)
            gen_files=[f for f in dir_list if f.startswith("gen")]
            dis_files=[f for f in dir_list if f.startswith("dis")]
            gen_files.sort()
            dis_files.sort()

            generator.load_state_dict(torch.load(cfg.weights_dir+"/"+gen_files[-1]))
           
            print(f"loaded weights from {cfg.weights_dir}/{gen_files[-1]} and {cfg.weights_dir}/{dis_files[-1]}")
    else:
        exit("No weights found")

else:
    generator.load_state_dict(torch.load(args.weights_path))
    print(f"loaded weights from {args.weights_path}")

    
cfg.device = torch.device("cpu")    
generator.to(cfg.device)
generator.eval()


def norm(img):
            low=float(img.min())
            high=float(img.max())
            img.sub_(low).div_(max(high - low, 1e-5))

# transforms = vt.Compose([vt.ToTensor(),vt.Normalize(0.5, 0.5),
#     vt.Resize((cfg.imsize, cfg.imsize),antialias=True)])
transforms = vt.Compose([vt.ToTensor(),vt.Resize((cfg.imsize, cfg.imsize),antialias=True)])

dataset = ImageFolder(
    root=cfg.data_dir, transform=transforms
)
def recover_image(tensor):
        tensor=tensor.detach().cpu().numpy().transpose(1, 2,0)*255
        
        return Image.fromarray(tensor.astype(np.uint8))
    
dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)
itr=iter(dataloader)
fid=FrechetInceptionDistance(feature=2048,normalize=True)
efid=metrics.FrechetInceptionDistance()
for i in tqdm(range(cfg.num_sample_epochs)):
         real_images=next(itr)[0].float().to(cfg.device)
         noise = random_sample(cfg.batch_size,cfg.z_dim,cfg.device)
         fake_images = generator(noise)
         norm(fake_images)
         norm(real_images)
         efid.update(real_images,is_real=True)
         efid.update(fake_images,is_real=False)
         fid.update(real_images,real=True)
         fid.update(fake_images,real=False)
print(fid.compute())
print(efid.compute())