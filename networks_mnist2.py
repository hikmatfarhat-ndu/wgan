import torch
import torch.nn as nn

# The networks are taken from 
# https://arxiv.org/abs/1511.06434

class Generator(nn.Module):

    def __init__(
        self,
        z_dim=100,
        out_ch=3,norm_type:str="batch",
        final_activation=None
    ):
        super().__init__()
        self.z_dim = z_dim
        self.out_ch = out_ch
        self.final_activation=final_activation

        self.net = nn.Sequential(
            Reshape(-1, self.z_dim),
            nn.Linear(self.z_dim, 128 * 7 * 7, bias=False),
             nn.LeakyReLU(0.2),
             Reshape(-1, 128, 7, 7),
            # * output: 7x7
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(128,norm_type),
            nn.LeakyReLU(0.2),
            # * output: 14x14
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(128,norm_type),
            nn.LeakyReLU(0.2),
            # * output: 28x28
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=3, bias=False),
                    
            )

    def forward(self, x):
        x = self.net(x)
        return x if self.final_activation is None else self.final_activation(x)
        #return torch.tanh(x)
      

class Discriminator(nn.Module):
    def __init__(self, in_ch=3,norm_type:str="batch",final_activation=None):
        super().__init__()
        self.in_ch = in_ch
        self.final_activation=final_activation
        self.net = nn.Sequential(
            # * 28x28
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(64,norm_type),
            nn.LeakyReLU(0.2),
            # * 14x14
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(64,norm_type),
            nn.LeakyReLU(0.2),
            # * 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1, bias=False),
        )

    def forward(self, x):
        x = self.net(x)
        return x if self.final_activation is None else self.final_activation(x)
    
class norm_layer(nn.Module):
    def __init__(self, num_channels,norm_type: str = None):
        super().__init__()
        if norm_type == "BatchNorm2d":
            self.norm = nn.BatchNorm2d(num_channels)
        elif norm_type == "GroupNorm":
            self.norm = nn.GroupNorm(num_channels, num_channels)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        #self.norm = getattr(torch.nn,norm_type)
                
    def forward(self, x):
        return x if self.norm is None else self.norm(x)
    

class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)