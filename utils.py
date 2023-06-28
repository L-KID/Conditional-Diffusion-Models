import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_images(images):
    """Plot images in a single figure"""
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    """Save images as a single figure"""
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    """Get training and validation/test dataset"""
    dataset_train = torchvision.datasets.FashionMNIST(
        root = './data/FashionMNIST',
        train = True,
        download = True,
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                    torchvision.transforms.Normalize((0.5, ), (0.5, ))]),
    ) 

    random_indices = torch.randperm(len(dataset_train)).tolist()
    trainset = torch.utils.data.Subset(dataset_train, random_indices[:55000])
    valset = torch.utils.data.Subset(dataset_train, random_indices[55000:])

    dataloader_train = torch.utils.data.DataLoader(trainset, batch_size=24)
    dataloader_val = torch.utils.data.DataLoader(valset, batch_size=1)

    if args.dataset == "train":
        return dataloader_train
    else:
        return dataloader_val


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
