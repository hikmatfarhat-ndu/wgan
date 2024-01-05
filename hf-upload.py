from first_model.model import WGAN_GP
from first_model.config import Config
import yaml
from munch import DefaultMunch
from transformers import AutoConfig,AutoModel
#from dcgan import Generator 
import torch
## two lines below are necessary to 
## upload the python code to huggingface hub
## i.e, config.py and model.py
## also, they are essential to load the model using AutoModel.from_pretrained("hikmatfarhat/wgan-gp")
Config.register_for_auto_class()
WGAN_GP.register_for_auto_class("AutoModel")

## not sure what the 2 lines below do
# AutoConfig.register("WGAN_GP",Config)
# AutoModel.register(Config, WGAN_GP)
cfg_path = "config.yml"
with open(cfg_path, "r") as f:
    print(f"Loading config file: {cfg_path}")
    cfg = yaml.safe_load(f)
cfg = DefaultMunch.fromDict(cfg)

config=Config(**cfg.toDict())
#config.save_pretrained("WGAN-GP")
model=WGAN_GP(config)
model.generator.load_state_dict(torch.load("/home/user/wgan-gp/chkpt/generator_499.pth"))
#model.save_pretrained("WGAN-GP")
model.push_to_hub(cfg.name)
# later on the command line use
# repo-id is the name of the repo, i.e. cfg.name
# huggingface-cli upload repo-id dcgan.py
# huggingface-cli upload "repo-id utils.py 

# To use the pretrained model, use the following code
#from transformers import AutoModel
# example repo-id "hikmatfarhat/wgan-gp"
#generator=AutoModel.from_pretrained(repo-id,trust_remote_code=True)