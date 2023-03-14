import torch
import torch.nn as nn
from torch.nn import functional as F

class VAE(nn.Module):

    def __init__(self, latent_dim: int, kl_loss_weight: float):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        self.kl_loss_weight = kl_loss_weight

        # initialize layers
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.deconv1 = torch.nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv5 = torch.nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv6 = torch.nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, stride=2, output_padding=1)


        # initialize weights
        # this is supposed to help with the vanishing gradient problem,
        # but I'd like to experiment more with why this is working
        nn.init.uniform_(self.conv1.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.conv2.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.conv3.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.conv4.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.conv5.weight, a=-0.05, b=0.05)

        nn.init.uniform_(self.deconv1.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.deconv2.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.deconv3.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.deconv4.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.deconv5.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.deconv6.weight, a=-0.05, b=0.05)


        ## encoder layers ##
        self.encoder = torch.nn.Sequential(
            self.conv1,
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            self.conv2,
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            self.conv3,
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            self.conv4,
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            self.conv5,
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(),
            # torch.nn.ReLU(),
            # nn.Flatten(),
            # torch.nn.Linear(3136, 2)
        )

        # latent space
        self.fc_mu = nn.Linear(512 * 4, latent_dim)
        self.fc_var = nn.Linear(512 * 4, latent_dim)

        # initialize weights to prevent divergence of KL in loss
        nn.init.uniform_(self.fc_mu.weight, a=-0.05, b=0.05)
        nn.init.uniform_(self.fc_var.weight, a=-0.05, b=0.05)

        self.decoder_start = nn.Linear(latent_dim, 512 * 4)
        nn.init.uniform_(self.decoder_start.weight, a=-0.05, b=0.05)

        # decoder layers
        self.decoder = torch.nn.Sequential(
            self.deconv6,
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(),
            self.deconv5,
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            self.deconv4,
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            self.deconv3,
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            self.deconv2,
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            self.deconv1,
            torch.nn.Sigmoid()
        )

    def encode(self, input):
        res = self.encoder(input)
        res = torch.flatten(res, start_dim=1)

        mu = self.fc_mu(res)
        log_var = self.fc_var(res)

        return {mu, log_var}

    def decode(self, z):
        res = self.decoder_start(z)
        res = res.view(-1, 512, 2, 2)
        res = self.decoder(res)

        return res


    def normalize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # convert log variance to variance
        eps = torch.randn_like(std)

        return eps * std + mu # convert to N(mu, std)

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.normalize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        exp_var = torch.exp(log_var)
        # exp_var= torch.clip(torch.exp(log_var), min=1e-5)

        kld_weight = self.kl_loss_weight # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - exp_var, dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}
