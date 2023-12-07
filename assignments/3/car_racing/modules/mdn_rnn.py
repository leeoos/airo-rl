import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions.normal import Normal

import gymnasium as gym
import numpy as np

from modules.mdn import MDN


class MDN_RNN(nn.Module):

    def __init__(self, input_size, output_size, mdn_units=512, hidden_size=256, num_mixs=5):
        super(MDN_RNN, self).__init__()

        self.name = 'MDR_RNN'
        self.hidden_size = hidden_size
        self.num_mixs = num_mixs
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.mdn = MDN(hidden_size, output_size, num_mixs, mdn_units)

    def forward(self, x, state=None):
        y = None
        x = x.unsqueeze(0) # batch first

        if state is None: y, state = self.lstm(x)
        else: y, state = self.lstm(x, state)

        pi, sigma, mu = self.mdn(y)
        # return state, sigma, mu
        return state, mu, None, sigma, pi
            
    def forward_lstm(self, x, state=None):
        y = None
        x = x.unsqueeze(0) # batch first

        if state is None: y, state = self.lstm(x)
        else: y, state = self.lstm(x, state)
        
        return y, state

    def loss(self, out, y, mu, logvar, sigma, pi, lambda_=1):
        # KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # return KL
        mdn_loss = self.mdn.loss(y, pi, mu, sigma)
        return mdn_loss

    def get_hidden_size(self):
        return self.hidden_size
    
    def save(self):
        torch.save(self.state_dict(), './models/'+self.name.lower()+'.pt')

    def load(self): 
        self.load_state_dict(torch.load('./models/'+self.name.lower()+'.pt', map_location=self.device))

    
    