import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions.normal import Normal

import gymnasium as gym
from modules.mdn import MDN

    
class MDN_RNN(nn.Module):

    def __init__(self, input_size, output_size, mdn_units=512, hidden_size=256, num_mixs=5):
        super(MDN_RNN, self).__init__()

        self.name = 'MDR_RNN'
        self.hidden_size = hidden_size
        self.num_mixs = num_mixs
        self.input_size = input_size
        self.output_size = output_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, 1, batch_first=True)
        self.mdn = MDN(hidden_size, output_size, num_mixs, mdn_units)

    def forward(self, x, state=None):
        
        y = None
        x = x.unsqueeze(0) # batch first
        if state is None: y, state = self.lstm(x)
        else: y, state = self.lstm(x, state)
        
        pi, sigma, mu = self.mdn(y)
        
        return state, sigma, mu
            
    def forward_lstm(self, x, state=None):
        
        y = None
        x = x.unsqueeze(0) # batch first
        if state is None:
            y, state = self.lstm(x)
        else:
            y, state = self.lstm(x, state)

        return y, state

    def loss_function(self, out, y, mu, logvar):
        #BCE = F.binary_cross_entropy(out, y, reduction="sum")
        KL = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return KL

    def get_hidden_size(self):
        return self.hidden_size
    