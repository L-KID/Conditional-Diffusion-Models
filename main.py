import argparse
from conditional_DDPM import train


def main():
    """
    :param args.run_name: Subfolder name for saving files
    :param args.epochs: The number of training epochs
    :param args.batch_size: The batch size for training
    :param args.image_size: The input image size
    :param args.num_classes: The number of dataset classes
    :param args.device: Specify the use of GPU or CPU
    :param args.lr: Set the learning rate
    :param args.dataset: Specify whether the training or validation set will be loaded
    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_conditional"
    args.epochs = 6
    args.batch_size = 14
    args.image_size = 28
    args.num_classes = 10
    args.device = "cpu"
    args.lr = 3e-4
    args.dataset = "train"


    train(args)


if __name__ == '__main__':
    main()