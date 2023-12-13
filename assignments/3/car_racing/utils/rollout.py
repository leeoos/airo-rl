import torch
from torchvision import transforms

import math
import numpy as np
import gymnasium as gym

from tqdm import tqdm
from time import sleep


class Rollout():

    def __init__(self):
        pass

    # ctallec functions:
    def unflatten_parameters(self, params, example, device):
        """ Unflatten parameters. Note: example is generator of parameters (module.parameters()), used to reshape params """
        
        params = torch.Tensor(params).to(device)
        idx = 0
        unflattened = []
        for e_p in example:
            unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
            idx += e_p.numel()
        return unflattened
    

    # def load_parameters(self, params, controller):
    #     """ Load flattened parameters into controller.

    #     :args params: parameters as a single 1D np array
    #     :args controller: module in which params is loaded
    #     """
    #     proto = next(controller.parameters())
    #     params = self.unflatten_parameters(params, controller.parameters(), proto.device)

    #     for p, p_0 in zip(controller.parameters(), params):
    #         p.data.copy_(p_0)
    

    def rollout(self, 
                agent, 
                params=None, 
                limit=1000, 
                device='cpu', 
                render=False, 
                continuous=False, 
                temperature=1000
        ):
       

        render_mode = 'human' if render else 'rgb_array'
        self.env = gym.make('CarRacing-v2', continuous=continuous, render_mode=render_mode)

        if params is not None:
            params = self.unflatten_parameters(params, agent.c.parameters(), device)

            # load parameters into agent controller
            for p, p_0 in zip(agent.c.parameters(), params):
                p.data.copy_(p_0)

        # env params
        obs, _ = self.env.reset()
        cumulative = 0
        done = False

        gamma = 0.9
        t = 0

        for _ in range(limit):
            action = agent.act(obs) 
            obs, reward, terminated, truncated, _ = self.env.step(action)
                
            done = terminated 
            if done: break

            # cumulative += math.exp(-t) * reward # 50 100 -50
            cumulative += reward # 50 100 -50
            t += 1

        self.env.reset()
        
        # print("cumulative: {}".format(cumulative))
        weighted_reward = cumulative + temperature # reward "temperature"
        return (- weighted_reward),  cumulative# 950 900 1050 
    

