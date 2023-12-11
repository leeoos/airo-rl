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
        self.load_state_dict(torch.load(dir+self.name.lower()+'.pt', map_location=self.device))
    


# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.conv1 = nn.Conv2d(CHANNELS, 32, 4, stride=2)  # Output: [32, 47, 47]
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2) # Output: [64, 22, 22]
#         self.conv3 = nn.Conv2d(64, 128, 4, stride=2) # Output: [128, 10, 10]
#         self.conv4 = nn.Conv2d(128, 256, 4, stride=2) # Output: [256, 4, 4]
#         self.fc_mu = nn.Linear(256*4*4, LATENT)
#         self.fc_logvar = nn.Linear(256*4*4, LATENT)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = x.view(x.size(0), -1) # Flatten the output
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         return mu, logvar

# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.fc = nn.Linear(LATENT, 256*4*4)
#         self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, output_padding=1)
#         self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, output_padding=1)
#         self.deconv4 = nn.ConvTranspose2d(32, CHANNELS, 4, stride=2, output_padding=1)

#     def forward(self, z):
#         z = F.relu(self.fc(z))
#         z = z.view(-1, 256, 4, 4) # Un-flatten
#         z = F.relu(self.deconv1(z))
#         z = F.relu(self.deconv2(z))
#         z = F.relu(self.deconv3(z))
#         reconstruction = torch.sigmoid(self.deconv4(z))
#         return reconstruction

# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()

#     def forward(self, x):
#         mu, logvar = self.encoder(x)
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         z = mu + eps * std
#         return self.decoder(z), mu, logvar
    
#     def set_device(self, device):
#         self.device = device
    
#     def save(self, dest):
#         if not exists(dest): mkdir(dest)
#         else: 
#             if exists(dest+self.name.lower()+'.pt'):
#                 remove(dest+self.name.lower()+'.pt')
#         torch.save(self.state_dict(), dest+self.name.lower()+'.pt')

#     def load(self, dir): 
#         self.load_state_dict(torch.load(dir+self.name.lower()+'.pt', map_location=self.device))

# class VAE(nn.Module):
#     """ Variational Autoencoder (specific for this task)"""

#     def __init__(self):
#         super().__init__()

#         # global variables
#         self.name = 'VAE'
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(CHANNELS, 16, kernel_size=3, stride=2),  # Convolution with stride 1
#             nn.LeakyReLU(0.2),
#             nn.AvgPool2d(kernel_size=4, stride=2),  # Average pooling to reduce dimensions
#             nn.Flatten(),
#             nn.Linear(16*22*22, 512),  # Adjusted for new flattened size
#             nn.LeakyReLU(0.2),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2)
#         )

#         # latent
#         self.mean_layer = nn.Linear(256, LATENT)
#         self.logvar_layer = nn.Linear(256, LATENT)

#         # decoder
#         self.fc1 = nn.Sequential(
#             nn.Linear(32, 288),
#             nn.LeakyReLU(0.2)
#         )
#         self.upsample = nn.Upsample(scale_factor=(1, 96))  # Upsampling to increase spatial dimensions
#         self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1) 


#     def encode(self, x):
#         x = self.encoder(x)
#         mu = self.mean_layer(x)
#         logsigma =  self.logvar_layer(x)
#         return mu, logsigma
    
#     def decode(self, x):
#         x = self.fc1(x)
#         x = x.view(-1, 3, 96)  # Reshape to (1, 3, 96)
#         x = self.upsample(x.unsqueeze(-1))  # Upsample to (1, 3, 96, 96)
#         x = F.leaky_relu(self.conv(x))
#         return x

#     def latent(self, mu, logsigma):
#         eps = torch.randn_like(logsigma).to(self.device)      
#         z = eps.mul(logsigma).add_(mu)
#         return z

#     def forward(self, x):
#         mu, logsigma = self.encode(x)
#         z = self.latent(mu, logsigma)
#         recon_x = self.decode(z)
#         return recon_x, mu, logsigma

 

# class VAE(nn.Module):
#     """ Variational Autoencoder """

#     def __init__(self, img_channels, LATENT):
#         super().__init__()

#         # global variables
#         self.name = 'VAE'
#         self.img_channels = img_channels
#         self.LATENT = LATENT
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         # encoder
#         self.conv1 = nn.Conv2d(img_channels, 32, 4, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
#         self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
#         self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
#         self.fc_mu = nn.Linear(2*2*256, LATENT)
#         self.fc_logsigma = nn.Linear(2*2*256, LATENT)

#         # decoder
#         self.fc1 = nn.Linear(LATENT, 1024)
#         self.deconv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
#         self.deconv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
#         self.deconv4 = nn.ConvTranspose2d(32, img_channels, 6, stride=2)


#     def encode(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = x.view(x.size(0), -1)

#         mu = self.fc_mu(x)
#         logsigma = self.fc_logsigma(x)
#         return mu, logsigma
    
#     def decode(self, x):
#         x = F.relu(self.fc1(x))
#         x = x.unsqueeze(-1).unsqueeze(-1)
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = F.relu(self.deconv3(x))

#         reconstruction = F.sigmoid(self.deconv4(x))
#         return reconstruction

#     def latent(self, mu, logsigma):
#         sigma = logsigma.exp()
#         eps = torch.randn_like(sigma)
#         z = eps.mul(sigma).add_(mu)
#         return z

#     def forward(self, x):
#         mu, logsigma = self.encode(x)
#         z = self.latent(mu, logsigma)
#         recon_x = self.decode(z)
#         return recon_x, mu, logsigma

#     def set_device(self, device):
#         self.device = device
    
#     def save(self, dest):
#         if not exists(dest): mkdir(dest)
#         else: 
#             if exists(dest+self.name.lower()+'.pt'):
#                 remove(dest+self.name.lower()+'.pt')
#         torch.save(self.state_dict(), dest+self.name.lower()+'.pt')

#     def load(self, dir): 
#         self.load_state_dict(torch.load(dir+self.name.lower()+'.pt', map_location=self.device))
