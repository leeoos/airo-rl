import torch
import torch.nn as nn
import torch.nn.functional as F

from os import mkdir, unlink, listdir, getpid, remove
from os.path import join, exists

LATENT = 32
CHANNELS = 3
OBS_SIZE = 96

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 32, 4, stride=2)  # Output: [32, 47, 47]
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2) # Output: [64, 22, 22]
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2) # Output: [128, 10, 10]
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2) # Output: [256, 4, 4]
        self.fc_mu = nn.Linear(256*4*4, LATENT)
        self.fc_logvar = nn.Linear(256*4*4, LATENT)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(x.size(0), -1) # Flatten the output
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(LATENT, 256*6*6)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # Output: 12x12
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # Output: 24x24
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)    # Output: 48x48
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1)     # Output: 96x96

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = z.view(-1, 256, 6, 6)  # Un-flatten to 6x6
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        reconstruction = torch.sigmoid(self.deconv4(z))  # Output shape [batch_size, 3, 96, 96]
        return reconstruction

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.name = 'VAE2'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return self.decoder(z), mu, logvar
    
    def get_latent(self, x):
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return(z)
    
    def save(self, dest):
        if not exists(dest): mkdir(dest)
        else: 
            if exists(dest+self.name.lower()+'.pt'):
                remove(dest+self.name.lower()+'.pt')
        torch.save(self.state_dict(), dest+self.name.lower()+'.pt')

    def load(self, dir): 
        if exists(dir+self.name.lower()+'.pt'):
            print("Loading model "+self.name+" state parameters")
            self.load_state_dict(torch.load(dir+self.name.lower()+'.pt', map_location=self.device))
            return self
        else:
            print("Error no model "+self.name.lower()+" found!")
            exit(1)