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

        # discrete
        self.fc = nn.Linear(LATENT, ACTIONS)
        self.th = nn.Tanh()

    def forward(self, c_in):
        out = self.fc(c_in)
        # out = self.th(out)
        return out
    
    def save(self, dest, f_value=None):
        if not exists(dest): mkdir(dest)
        else: 
            if exists(dest+self.name.lower()+'.pt'):
                remove(dest+self.name.lower()+'.pt')
        torch.save(
            {
                'state_dict': self.state_dict(),
                'f_value': f_value
            }, dest+self.name.lower()+'.pt'
        )

    def load(self, dir, get_value=None): 
        if exists(dir+self.name.lower()+'.pt'):
            print("Loading model "+self.name+" state parameters")
            state = torch.load(dir+self.name.lower()+'.pt', map_location={'cuda:0': 'cpu'})
            self.load_state_dict(state['state_dict'])

            if get_value : 
                return state['f_value']
            return self
        
        else:
            print("Error no model "+self.name.lower()+" found!")
            exit(1)