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
        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='robot')
        self.env.reset()

        # nn variables
        self.latent_dim = 100
        self.hidden_dim = 256
        self.action_space_dim = 3
        
        # global variables
        self.device = device
        self.batch_size = 32
        self.num_rollout = 1
        self.starting_state = True
        self.memory = None # rnn state is empy at the beginning
       
        # models
        self.vae = VAE(latent_size=self.latent_dim)
        self.rnn = MDN_RNN(
            input_size = self.latent_dim + self.action_space_dim, 
            output_size = self.latent_dim + self.action_space_dim
        )
        self.c = nn.Linear(
            in_features= self.latent_dim + self.hidden_dim, 
            out_features= self.action_space_dim
        )


    def forward(self, x):

        if self.starting_state:
            h = torch.zeros(1, self.hidden_dim)
            self.starting_state = False

        else: h = self.memory
        # print("h: ", h.shape)

        mu, logvar = self.vae.encode(x.float())
        z = self.vae.latent(mu, logvar)
        # print("z: ", z.shape)

        c_in = torch.cat((z, h),dim=1).to(self.device)
        # print("c_in: ", c_in.shape)

        
        return c_in, z, h
    
    def act(self, state):
        c_in, z, h = self.forward(state)
        a = self.c(c_in).to(self.device)
        # print("action: ", a.shape)

        rnn_in = torch.concat((a, z), dim=1).to(self.device)
        # print("rnn_in: ", rnn_in.shape)

        self.memory, _ = self.rnn.forward_lstm(rnn_in)
        self.memory = self.memory.squeeze(0)
        # print("memory: ", self.memory.shape)
        
        torch.clip(a, min = -1, max = 1 )
        return a.cpu().float().squeeze().detach().numpy()

    def train(self):
        # random rollout to collect observations
        rollout_obs, rollout_actions = Rollout(self.env).random_rollout(self.num_rollout)

        # train the vae
        self.vae = Trainer().train(
            model_ =self.vae, 
            data_=rollout_obs, 
            batch_size_=self.batch_size,
            retrain=False
        )

        # encode the observation to train the rnn
        mu, logvar = self.vae.encode(rollout_obs)
        rollout_latent = self.vae.latent(mu, logvar)
        rollout_al = torch.cat((rollout_actions, rollout_latent), dim=1)

        # print(self.rnn.forward_lstm(rollout_al)[0].shape)

        # train the rnn
        self.rnn = Trainer().train(
            model_ =self.rnn, 
            data_=rollout_al.detach(), 
            batch_size_=self.batch_size,
            retrain=False
        )

        print(Rollout(self.env).rollout(self, self.c))

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
