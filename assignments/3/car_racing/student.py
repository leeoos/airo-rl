import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor, List
from torch.utils.data import Dataset
from torch.distributions.normal import Normal

import gymnasium as gym
import numpy as np

from utils.rollout import Rollout
from modules.vae import VAE
from modules.mdrnn import MDN_RNN
import cma


class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device

        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        self.env.reset()

        # Local variables
        latent_dim = 100
        hidden_dim = 100
        action_space_dim = 3
       
        # Models
        self.vae = VAE(latent_size=latent_dim)
        self.rnn = MDN_RNN(input_size=latent_dim, output_size=hidden_dim)
        self.c = self.c = nn.Linear(latent_dim+hidden_dim, action_space_dim)

    def forward(self, x):
        h = self.rnn.initial_state()
        z = self.vae.encode(x)

        return [z,h]
    
    def act(self, state):
        input = self.forward(state)
        a = self.c(input)
       
        return a

    def train(self):

        # Train the Variational Autoencoder
        rollout_obs, rollout_actions = Rollout(self.env).random_rollout(num_rollout=32)
        self.train_nn(
            network=self.vae, 
            loss_fn=self.vae.loss_function,
            optimizer=torch.optim.Adam(self.vae.parameters(), lr=0.001), 
            data=rollout_obs, 
            batch_size=32, 
            num_epochs=100
        )

        # Train the MDN RNN
        mu, logvar = self.vae.encode(rollout_obs)
        rollout_latent = self.vae.latent(mu, logvar)
        rollout_mix = torch.cat((rollout_actions, rollout_latent), dim=1)

        self.train_nn(
            network=self.rnn, 
            loss_fn=self.rnn.loss_function,
            optimizer=torch.optim.Adam(self.rnn.parameters(), lr=0.001), 
            data=rollout_mix, 
            batch_size=32, 
            num_epochs=100
        )

    def save(self):
        ...
        # torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        ...
        # self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device

        return ret

    def train_nn(self, network, loss_fn, optimizer, data, batch_size, num_epochs):
    
        for epoch in range(num_epochs):
            for i in range(0, len(data), batch_size):

                # Get batch
                X_batch = data[i:i+batch_size]
                y_batch = data[i:i+batch_size]

                # Forward pass
                loss = 0
                if network.name == 'VAE': 
                    outputs, mu, sigma = network.forward(X_batch)
                    loss = loss_fn(outputs, y_batch, mu, sigma)

                # elif network.name == 'MDN_RNN':
                else:
                    outputs, mu, sigma = network.forward(X_batch.detach())
                    loss = loss_fn(outputs, y_batch, mu, sigma)
                
                # print(f'a{epoch, i}')
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                # print(optimizer.state_dict)
                optimizer.step()
                # print(f'b{epoch, i}')

            # Print the loss every 100 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Model_{network.name}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')