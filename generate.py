import torch
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from modules import UNet_conditional
from conditional_DDPM import Diffusion
from utils import *


device = "cpu"
model = UNet_conditional(c_in=1, c_out=1, num_classes=10).to(device)
ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=28, device=device)
# The number of sampling
n = 2

# Get a target image from the validation set, and its corresponding label as guidance
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.run_name = "DDPM_conditional"
args.dataset = "val"
dataloader = get_data(args)
example_images, example_labels = next(iter(dataloader))
example_images = (example_images.clamp(-1, 1) + 1) / 2
example_images = (example_images * 255).type(torch.uint8)
y = example_labels.to(device)
# Generate images corresponding to label y
x = diffusion.sample(model, n, y, cfg_scale=0)
plot_images(x)
save_images(x, os.path.join("results", "generated_img", "generated.jpg"))
save_images(example_images, os.path.join("results", "generated_img", "target.jpg"))
