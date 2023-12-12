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
        pass

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
    

    def rollout(self, agent, params=None, limit=1000, device='cpu', render=False):
        """ Execute a rollout and returns minus cumulative reward. """

        render_mode = 'human' if render else 'rgb_array'
        self.env = gym.make('CarRacing-v2', continuous=False, render_mode=render_mode)

        if params is not None:
            params = self.unflatten_parameters(params, agent.c.parameters(), device)

            # load parameters into agent controller
            for p, p_0 in zip(agent.c.parameters(), params):
                p.data.copy_(p_0)

        # ####DEGUB####
        # for p in agent.c.parameters():
        #     print('new parameters: {}'.format(p))
        #     break
        # ####DEGUB####

        obs, _ = self.env.reset()
        cumulative = 0
        done = False

        for _ in range(limit):
            action = agent.act(obs) 
            obs, reward, terminated, truncated, _ = self.env.step(action)
                
            done = terminated 
            if done: break

            cumulative += reward # 50 100 -50

        # print("cumulative: {}".format(cumulative))
        cumulative += 1000
        return (- cumulative) # 950 900 1050 
    
    # if cum > 1000 --> res: negativo (piccolo) --> non fare niente
    # if cum < -1000 --> res: positivo (grande) --> minimizza

    # best: negativo (piccolo) -->  cur_best > best --> aggiorna cur_best
    # best: positivo (grande) --> cur_best 


    # - (950 - cum) >= 0
    
    # cum >= 950

