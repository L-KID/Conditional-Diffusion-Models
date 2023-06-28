import torch
from torchvision.utils import save_image
from conditional_DDPM import Diffusion
from utils import get_data
import argparse


# Test the noise_images method
parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  
args.image_size = 28
args.dataset_path = 'train'

dataloader = get_data(args)

diff = Diffusion(device="cpu")

example_images, example_labels = next(iter(dataloader))
t = torch.Tensor([50, 100, 150, 200, 300, 600, 700, 999]).long()

noised_image, _ = diff.noise_images(example_images, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
