import torch
import torch.nn as nn
torch.manual_seed(42)


from modules.vae import LATENT, OBS_SIZE

from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, remove

ACTIONS = 4
CONTINUOUS = 3

class Continuous(nn.Module):
    """ Controller """

    def __init__(self):
        super().__init__()

        self.name = 'CONTINUOUS'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # continuous
        self.fc= nn.Linear(LATENT, CONTINUOUS)
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