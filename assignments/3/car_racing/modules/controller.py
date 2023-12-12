import torch
import torch.nn as nn

from modules.vae import LATENT, OBS_SIZE

from os import mkdir, unlink, listdir, getpid, remove
from os.path import join, exists

ACTIONS = 4

class Controller(nn.Module):
    """ Controller """

    def __init__(self):
        super().__init__()

        self.name = 'CONTROLLER'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.fc = nn.Linear(LATENT, ACTIONS)
        self.th = nn.Tanh()

    def forward(self, c_in):
        out = self.fc(c_in)
        # out = self.th(out)
        return out
    
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