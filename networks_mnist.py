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
            nn.ConvTranspose2d(self.z_dim, 128, kernel_size=4, stride=1, padding=0, bias=False),
            norm_layer(128,norm_type),
            nn.ReLU(),
            # * Layer 2: 4x4
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(64,norm_type),
            nn.ReLU(),
            # * Layer 3: 8x8
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(32,norm_type),
            nn.ReLU(),
            # * Layer 4: 16x16
            nn.ConvTranspose2d(32, self.out_ch, kernel_size=4, stride=2, padding=3, bias=False),
            # * 28x28
        )

    def forward(self, x):
        x = self.net(x)
        return torch.tanh(x)
      

class Discriminator(nn.Module):
    def __init__(self, in_ch=3,norm_type:str="batch"):
        super().__init__()
        self.in_ch = in_ch

        self.net = nn.Sequential(
            # * 28x28
            nn.Conv2d(self.in_ch, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            # * 14x14
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(64,norm_type),
            nn.LeakyReLU(0.2),
            # * 7x7
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32,norm_type),
            nn.LeakyReLU(0.2),
            # * 4x4 
            nn.Conv2d(32, 1, kernel_size=4, stride=1, padding=0, bias=False)
            # * 1x1
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