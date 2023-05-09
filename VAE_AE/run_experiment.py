"""
Script to run an experiment with parameters from a configuration (.yaml) file
"""
import torch
import argparse
import yaml

from torchvision import transforms
from torch.utils.data import DataLoader
from masked_celeba import CelebA
from models.VAE import VAE
from VAEExperiment import VAEExperiment

# set up argument parser for configuration file
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae_base_config.yaml')


args = parser.parse_args()
# read config file
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# get device to train on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Running on ', device)

# get data
data_root = config['data_root']

# Spatial size of training images, images are resized to this size.
image_size = config['img_size']

num_workers = config['num_workers']

# how many samples per batch to load
batch_size = config['batch_size']

num_epoch = config['num_epoch']

celeba_train = CelebA(data_root,
                              download=False,
                              split = "train",
                              transform=transforms.Compose([
                                  transforms.Grayscale(),
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                              ]),
                              proportion = 0.15)

celeba_val = CelebA(data_root,
                              download=False,
                              split = "valid",
                              transform=transforms.Compose([
                                  transforms.Grayscale(),
                                  transforms.Resize(image_size),
                                  transforms.CenterCrop(image_size),
                                  transforms.ToTensor(),
                              ]),
                              proportion = 0.15)

# prepare data loaders
train_loader = torch.utils.data.DataLoader(celeba_train, batch_size=batch_size, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(celeba_val, batch_size=batch_size, num_workers=num_workers)

# create dictionary of dataloaders to pass to experiment
dataloaders = {"train": train_loader,
                   "val": val_loader}

kl_loss_weight = batch_size / len(celeba_train)

# initialize model
model = VAE(latent_dim = config['latent_dim'], kl_loss_weight = kl_loss_weight)

model.to(device)

experiment = VAEExperiment(model, dataloaders, config['lr'], config['gamma'], device, config['name'], config['base_dir'], num_epoch=num_epoch)

experiment.train_model()

experiment.save_metrics()

experiment.save_model()

experiment.plot_training()

