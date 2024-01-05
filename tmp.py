from dcgan import Generator
import torch
import torch
g=Generator(64,3,128,"GroupNorm","tanh")
g.load_state_dict(torch.load("chkpt/generator_249.pth"))
torch.save(g,"g.pt")