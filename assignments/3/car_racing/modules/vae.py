import torch
import torch.nn as nn
import torch.nn.functional as F

import gymnasium as gym


class VAE(nn.Module):
    
    def __init__(self, latent_size):
        super().__init__()

        self.name = 'VAE'
        self.latent_size = latent_size
        self.device = None
        
        # encoder
        self.enc_conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.enc_conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(64)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # z
        self.mu = nn.Linear(1024, latent_size)
        self.logvar = nn.Linear(1024, latent_size)
        
        # decoder
        self.fc = nn.Linear(100, 6 * 6 * 256)  # Convert to 6x6x256 tensor
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)

        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.latent(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar
        
    def encode(self, x):
        batch_size = x.shape[0]
        
        out = F.relu(self.enc_conv1(x))
        out = F.relu(self.enc_conv2(out))
        out = F.relu(self.enc_conv3(out))
        # out = self.batch_norm(out)
        out = F.relu(self.adaptive_pool(out))
        out = out.reshape(batch_size,1024)
        
        mu = self.mu(out)
        logvar = self.logvar(out)
        
        return mu, logvar
        
    def decode(self, z):
        batch_size = z.shape[0]
        
        out = self.fc(z)
        out = out.view(-1, 256, 6, 6)
        out = F.relu(self.dec_conv1(out))
        out = F.relu(self.dec_conv2(out))
        out = torch.sigmoid(self.dec_conv3(out))
        out = torch.sigmoid(self.dec_conv4(out))
        
        return out
        
        
    def latent(self, mu, logvar):
        sigma = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar).to(self.device)
        z = mu + eps*sigma
        return z
    

    def set_device(self, device):
        self.device = device

    def loss_function(self, out, y, mu, logvar, lambda_=1):
        CE = F.cross_entropy(out, y)
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return lambda_*KL + CE
   
    # def obs_to_z(self, x):
    #     mu, logvar = self.encode(x)
    #     z = self.latent(mu, logvar)
    #     return z

    # def sample(self, z):
    #     out = self.decode(z)
    #     return out

    # def get_latent_size(self):
    #     return self.latent_size