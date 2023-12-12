import torch
import torch.nn as nn

from modules.vae import LATENT, OBS_SIZE

ACTIONS = 4

class Controller(nn.Module):
    """ Controller """

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT, ACTIONS)
        self.th = nn.Tanh()

    def forward(self, c_in):
        out = self.fc(c_in)
        # out = self.th(out)
        return out