import torch
import torch.nn as nn

# The networks are taken from 
# https://arxiv.org/abs/1511.06434

class Generator(nn.Module):
    #Outputs 64x64 pixel images

    def __init__(
        self,
        z_dim=100,
        out_ch=3,norm_type:str="batch"
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch


        self.net = nn.Sequential(
            # * Layer 1: 1x1
            nn.ConvTranspose2d(self.z_dim, 512, 4, 1, 0, bias=False),
            norm_layer(512,norm_type),
            nn.ReLU(),
            # * Layer 2: 4x4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            norm_layer(256,norm_type),
            nn.ReLU(),
            # * Layer 3: 8x8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            norm_layer(128,norm_type),
            nn.ReLU(),
            # * Layer 4: 16x16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            norm_layer(64,norm_type),
            nn.ReLU(),
            # * Layer 5: 32x32
            nn.ConvTranspose2d(64, self.out_ch, 4, 2, 1, bias=False),
            # * Output: 64x64
        )

    def forward(self, x):
        x = self.net(x)
        return torch.tanh(x)
      

class Discriminator(nn.Module):
    def __init__(self, in_ch=3,norm_type:str="batch"):
        super().__init__()
        self.in_ch = in_ch

        self.net = nn.Sequential(
            # * 64x64
            nn.Conv2d(self.in_ch, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # * 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            norm_layer(128,norm_type),
            nn.LeakyReLU(0.2),
            # * 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            norm_layer(256,norm_type),
            nn.LeakyReLU(0.2),
            # * 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            norm_layer(512,norm_type),
            nn.LeakyReLU(0.2),
            # * 4x4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        return x
class norm_layer(nn.Module):
    def __init__(self, num_channels,norm_type: str = "batch"):
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm2d(num_channels)
        elif norm_type == "group":
            self.norm = nn.GroupNorm(num_channels, num_channels)
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
    def forward(self, x):
        return self.norm(x)