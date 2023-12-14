import time
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Normalize, Resize,Compose
from torch.utils.data.dataloader import DataLoader
import yaml
from munch import DefaultMunch
from torch.utils.benchmark import Timer

cfg_path = "config.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)
print(cfg)
transforms = Compose([ToTensor(),Normalize(0.5, 0.5),
    Resize((cfg.imsize, cfg.imsize),antialias=True)])

dataset = ImageFolder(
    root=cfg.data_dir, transform=transforms
)
    



dataloader = DataLoader(
    dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    drop_last=True,
)
start = time.time()
count=0
for imgs,_ in dataloader:
    count+=1
    #imgs=imgs.to(cfg.device)
end = time.time()
print(end - start)
