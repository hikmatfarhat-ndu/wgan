import torch
import torch.nn as nn
from torch.optim import Adam
from torch import autograd
from tqdm import tqdm
from collections import defaultdict
import os
from PIL import Image
import numpy as np
from dcgan import Generator,Discriminator
from utils import init_weight,random_sample,norm
from torchvision.utils import make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
import Lightning as L

class LitWGAN_GP(L.LightningModule):
    """
    WGAN_GP Wasserstein GAN. Uses gradient penalty instead of gradient clipping to enforce 1-Lipschitz continuity. 
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_iter_per_g = 1 if self.cfg.d_iter_per_g is None else self.cfg.d_iter_per_g
# see https://arxiv.org/abs/1511.06434
        # self.generator = Generator(
        #     z_dim=self.cfg.z_dim,
        #     out_ch=self.cfg.img_ch,norm_type=self.cfg.g_norm_type,
        #     final_activation=self.cfg.g_final_activation
        # )
       
        self.generator=Generator(cfg.imsize,cfg.img_ch,cfg.zdim,
                                 norm_type=cfg.norm_type.g,
                                 final_activation=self.cfg.final_activation.g)
        self.discrim = Discriminator(cfg.imsize,cfg.img_ch,norm_type=cfg.norm_type.d,
                                     final_activation=cfg.final_activation.d)
    def config_optimizers(self):

        self.optG = Adam(self.generator.parameters(), lr=self.cfg.lr.g)
        self.optD = Adam(self.discrim.parameters(), lr=self.cfg.lr.d)
        return [self.optG,self.optD], []
    def generator_step(self, data):

        noise = random_sample(self.cfg.batch_size, self.cfg.zdim, self.cfg.device)
        fake_images = self.generator(noise)
        fake_logits = self.discrim(fake_images)
        g_loss = -fake_logits.mean().view(-1)
        self.optG.zero_grad()

        g_loss.backward()
        self.optG.step()

        self.metrics["G-loss"] += [g_loss.item()]

    def discriminator_step(self, data):
        
        real_images = data[0].float().to(self.cfg.device)
        noise = random_sample(self.cfg.batch_size, self.cfg.zdim, self.cfg.device)
        fake_images = self.generator(noise)
        
        real_logits = self.discrim(real_images)
        fake_logits = self.discrim(fake_images)

        gradient_penalty = self.cfg.w_gp * self._compute_gp(
            real_images, fake_images
        )

        loss_c = fake_logits.mean() - real_logits.mean()
        d_loss = loss_c + gradient_penalty

        self.optD.zero_grad()
        d_loss.backward()
        self.optD.step()

        self.metrics["D-loss"] += [d_loss.item()]
        self.metrics["GP"] += [gradient_penalty.item()]
    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs,_=batch
        if optimizer_idx==0:
            self.generator_step(imgs)
        if optimizer_idx==1:
            self.discriminator_step(imgs)
        return self.metrics