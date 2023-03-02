import torch
import torch.nn as nn

class BasicAEII(nn.Module):

    def __init__(self):
        super(BasicAEII, self).__init__()
        ## encoder layers ##
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding = 1, stride = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding = 1, stride = 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding = 1, stride = 2),
            # torch.nn.ReLU(),
            # nn.Flatten(),
            # torch.nn.Linear(3136, 2)
        )

        self.decoder = torch.nn.Sequential(
            # torch.nn.Linear(2, 3136),
            # Resize(-1, 64, 7,7),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
