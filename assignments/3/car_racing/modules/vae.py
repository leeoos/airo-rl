import torch
import torch.nn as nn
import torch.nn.functional as F

from os import mkdir, unlink, listdir, getpid, remove
from os.path import join, exists

LATENT = 32
CHANNELS = 3
OBS_SIZE = 96


class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.name = 'VAE'
        self.LATENT = LATENT
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # z
        self.mu = nn.Linear(1024, LATENT)
        self.logvar = nn.Linear(1024, LATENT)
        
        # decoder
        self.fc = nn.Linear(LATENT, 6 * 6 * 256)  # Convert 1024 elements back to 4x4x64 tensor
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar   
    
    def get_latent(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        return z
        
    def encode(self, x):
        self.batch_size = 1
        if len(x.shape)>3:
            self.batch_size = x.shape[0]
        
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.adaptive_pool(out))
        out = out.reshape(self.batch_size,1024)

        mu = self.mu(out)
        logvar = self.logvar(out)
        return mu, logvar
        
    def decode(self, z):
        out = self.fc(z)
        out = out.view(self.batch_size, 256, 6, 6)
    
        out = F.relu(self.dec_conv1(out))
        out = F.relu(self.dec_conv2(out))
        out = torch.sigmoid(self.dec_conv3(out))
        # out = F.relu(self.dec_conv3(out))
        out = torch.sigmoid(self.dec_conv4(out))
        return out
           
    def latent(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar).to(self.device)
        z = mu + eps*sigma
        return z
    
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
            

        
#     def decode(self, z):
#         out = self.fc(z)
#         out = out.view(self.batch_size, 256, 6, 6)

#         out = F.relu(self.dec_conv1(out))
#         out = F.relu(self.dec_conv2(out))
#         out = F.relu(self.dec_conv3(out))
#         out = torch.sigmoid(self.dec_conv4(out))
        
#         return out
        


