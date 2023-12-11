import torch
from torchvision import transforms
import gymnasium as gym
import numpy as np
import math
from tqdm import tqdm
from time import sleep

from modules.vae import LATENT, OBS_SIZE

class Rollout():

    def __init__(self):
        self.env = gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array')

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((OBS_SIZE, OBS_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    
    # ctallec function:
    def unflatten_parameters(self, params, example, device):
        """ Unflatten parameters. Note: example is generator of parameters (module.parameters()), used to reshape params """
        
        params = torch.Tensor(params).to(device)
        idx = 0
        unflattened = []
        for e_p in example:
            unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
            idx += e_p.numel()
        return unflattened
    
    # this function is here just to test cma parametrs 
    # def get_action(self, state, vae, controller, device):
    #     state = torch.tensor(state/255, dtype=torch.float)
    #     state = state.unsqueeze(0).to(device)
       
    #     mu, logvar = vae.encode(state.float())
    #     z = vae.latent(mu, logvar)
    #     a = controller(z).to(device)

    #     torch.clip(a, min = -1, max = 1 )
    #     return a.cpu().float().squeeze().detach().numpy()

    def rollout(self, agent, params=None, limit=1100, device='cpu', display=False):
        """ Execute a rollout and returns minus cumulative reward. """

        if params is not None:
            params = self.unflatten_parameters(params, agent.c.parameters(), device)

            # load parameters into agent controller
            for p, p_0 in zip(agent.c.parameters(), params):
                p.data.copy_(p_0)

        obs, _ = self.env.reset()
        cumulative = 0
        step_counter = 0 
        done = False

        for _ in range(limit):
            action = agent.act(obs) #self.get_action(obs, vae, controller, device) 
            obs, reward, terminated, truncated, _ = self.env.step(action)
                
            done = terminated 
            if done: break

            cumulative += reward

        if display: print(f"Total Reward: {cumulative}, steps: {step_counter}")
        return (-cumulative)

