import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor, List
from torch.utils.data import Dataset
from torch.distributions.normal import Normal

import gymnasium as gym
import numpy as np
import os 

from utils.rollout import Rollout
from utils.module_trainer import Trainer
from modules.vae import VAE
from modules.mdn_rnn import MDN_RNN
import cma

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()

        # gym env
        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')
        self.env.reset()

        # local variables
        latent_dim = 100
        hidden_dim = 256
        action_space_dim = 3
        
        # global variables
        self.device = device
        self.batch_size = 32
        # self.memory = torch.zeros()
       
        # models
        self.vae = VAE(latent_size=latent_dim)
        self.rnn = MDN_RNN(input_size=latent_dim+action_space_dim, output_size=latent_dim+action_space_dim)
        self.controller = nn.Linear(latent_dim+hidden_dim, action_space_dim)



    def forward(self, x):
        ...
    
    def act(self, state):
        mu, logvar = self.VAE.encode(state.float())
        z = self.VAE.latent(mu, logvar)

        def my_function():
            if not hasattr(my_function, "is_first_call"):
                my_function.is_first_call = True
            else:
                my_function.is_first_call = False

            if my_function.is_first_call:
                return torch.tensor([0,0,0]).to(self.device)
            else:
                return a

        rnn_in = torch.concat((a, z), dim=1).to(self.device)
        rnn_out, h = self.MDN_RNN.forward_lstm(rnn_in)
        
        a = self.C(c_in).to(self.device)
        torch.clip(a, min = -1, max = 1 )
        c_in = torch.cat((z, state),dim=1).to(self.device)

        return a

    def train(self):

        # uncomment to delate pre-trainde models parameters
        # os.remove('./models/vae.pt')
        # os.remove('./models/mdr_rnn.pt')

        # random rollout to collect observations
        rollout_obs, rollout_actions = Rollout(self.env).random_rollout(num_rollout=1)

        # train the vae
        self.vae = Trainer().train(
            model_ =self.vae, 
            data_=rollout_obs, 
            batch_size_=self.batch_size
        )

        # encode the observation to train the rnn
        mu, logvar = self.vae.encode(rollout_obs)
        rollout_latent = self.vae.latent(mu, logvar)
        rollout_al = torch.cat((rollout_actions, rollout_latent), dim=1)

        # print(self.rnn.forward_lstm(rollout_al)[0][0].shape)

        # train the rnn
        self.rnn = Trainer().train(
            model_ =self.rnn, 
            data_=rollout_al.detach(), 
            batch_size_=self.batch_size
        )

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
