import comet_ml
from comet_ml.integration.pytorch import load_model
from dcgan import Generator,Discriminator
generator=Generator(64,3,128,norm_type="GroupNorm",final_activation="tanh")
discriminator=Discriminator(64,3,norm_type="GroupNorm",final_activation=None)
checkpoint=load_model("registry://wgan/GAN-14:1.0.0")
#checkpoint=load_model("experiment://4e8543da67594b70b38ac622309f0ec4/GAN-9")
generator.load_state_dict(checkpoint["g_state_dict"])
discriminator.load_state_dict(checkpoint["d_state_dict"])