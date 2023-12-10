import torch


import gymnasium as gym
import numpy as np
import math
from tqdm import tqdm
from time import sleep

class Rollout():

    def __init__(self):
        pass
    
    def unflatten_parameters(self, params, example, device):
        """ Unflatten parameters. Note: example is generator of parameters (module.parameters()), used to reshape params """

        params = torch.Tensor(params).to(device)
        idx = 0
        unflattened = []
        for e_p in example:
            unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
            idx += e_p.numel()
        return unflattened

    def rollout(self, env, agent, controller, params=None, limit=2000, device='cpu', display=False):
        """ Execute a rollout and returns minus cumulative reward. """

        if params is not None:
            params = self.unflatten_parameters(params, controller.parameters(), device)

            # load parameters into controller
            for p, p_0 in zip(controller.parameters(), params):
                p.data.copy_(p_0)

        obs, _ = env.reset()
        cumulative = 0
        step_counter = 0 
        done = False

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            step_counter +=1
            done = terminated or (step_counter >= limit)

            cumulative += reward

        if display: print(f"Total Reward: {cumulative}, steps: {step_counter}")
        agent.starting_state = True
        return (-cumulative)

