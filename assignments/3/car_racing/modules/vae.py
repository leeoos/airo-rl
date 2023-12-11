import torch
import torch.nn as nn
import torch.nn.functional as F

from os import mkdir, unlink, listdir, getpid, remove
from os.path import join, exists

LATENT = 32
OBS_SIZE = 64

class VAE(nn.Module):
    """ Variational Autoencoder """

    def __init__(self, img_channels, latent_size):
        super().__init__()

        # global variables
        self.name = 'VAE'
        self.img_channels = img_channels
        self.latent_size = latent_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # encoder
        self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc_mu = nn.Linear(2*2*256, latent_size)
        self.fc_logsigma = nn.Linear(2*2*256, latent_size)

        # decoder
        self.fc1 = nn.Linear(latent_size, 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)


    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        mu = self.fc_mu(x)
        logsigma = self.fc_logsigma(x)
        return mu, logsigma
    
    def decode(self, x):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        reconstruction = F.sigmoid(self.deconv4(x))
        return reconstruction

    def latent(self, mu, logsigma):
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)
        return z

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.latent(mu, logsigma)
        recon_x = self.decode(z)
        return recon_x, mu, logsigma

    def set_device(self, device):
        self.device = device
    
    def save(self, dest):
        if not exists(dest): mkdir(dest)
        else: 
            if exists(dest+self.name.lower()+'.pt'):
                remove(dest+self.name.lower()+'.pt')
        torch.save(self.state_dict(), dest+self.name.lower()+'.pt')

    def load(self, dir): 
        self.load_state_dict(torch.load(dir+self.name.lower()+'.pt', map_location=self.device))