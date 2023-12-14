import torch
import torch.nn as nn
from torchvision import transforms
torch.manual_seed(42)

import sys
import time
from tqdm import tqdm
from time import sleep
from os.path import join, exists
from os import mkdir, remove, unlink, listdir, getpid

import numpy as np
import matplotlib.pyplot as plt

import cma
import numpy as np
import gymnasium as gym

# custom imports
from modules.controller import ACTIONS
from modules.controller import Controller

from modules.continuous import CONTINUOUS
from modules.continuous import Continuous

from modules.vae import VAE 
from modules.vae import LATENT, OBS_SIZE

# from modules.vae2 import VAE
# from modules.vae2 import LATENT, OBS_SIZE

from utils.rollout import Rollout
import train.train_vae as vae_trainer

class Policy(nn.Module):
    
    continuous = True

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()

        # gym env
        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='rgb_array')
        
        # global variables
        self.device = device
        self.modules_dir = './checkpoints/'
        self.data_dir = './dataset/'
       
        # models
        self.vae = VAE().to(self.device)

        self.c = None
        if not self.continuous:
            self.c = Controller().to(self.device)

        else:
            self.c = Continuous().to(self.device)

        # utils
        self.roller = Rollout()

        # cma training parameters
        self.pop_size = 6
        self.n_samples = 3
        self.fixed_seed = 588039 
        self.max_reward = 1000
        # self.stop_condiction = 700 # stop at (1000 - reward) e.g. s.c. = 200 --> reward = 800
        self.target_mean = 700 # target mean reward

  
    def act(self, state):

        # convert input state to a torch tensor
        state = torch.tensor(state/255, dtype=torch.float)
        state = state.permute(0,2,1).permute(1,0,2)
        state = state.unsqueeze(0).to(self.device)

        # obs compression
        z = self.vae.get_latent(state.float())
        
        # get action from controller
        a = self.c(z).to(self.device)   

        if not self.continuous:
            return (int(torch.argmax(a)) + 1)
        
        else:
            return a.cpu().float().squeeze().detach().numpy()


    def train(self):
        """Train the entire network or just the controller module"""

        # set to True to tarin vae and rnn
        train_vae = False
        train_rnn = False

        if train_vae: 
            if not exists(self.data_dir):
                print("Error: no data")
                exit(1)

            observations = torch.load(self.data_dir+'observations.pt')

            self.vae = vae_trainer.train_vae(
                model=self.vae,
                data=observations,
                epochs=10,
                device=self.device,
                save_dir=False
            ).to(self.device)

        else:
            self.vae = self.vae.load(self.modules_dir).to(self.device)

        # Train controller
        print("Attempting to load previous best...")
       
        cur_best = 100000000000 # max cap
        cur_mean = -100000000000 # min cap
        file_name = 'controller.pt' if not self.continuous else 'continuous.pt'
        if exists(self.modules_dir+file_name): 
            self.c = self.c.load(self.modules_dir)
            print("Previous controller loaded")
            cur_best = self.c.load(self.modules_dir, get_value=True)
            print("Best current value for the objective function: {}".format(cur_best))

        # set up cma parameters
        params = self.c.parameters()
        flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
        es = cma.CMAEvolutionStrategy(flat_params, 0.2, {'popsize':self.pop_size}) #'seed':self.fixed_seed

        # log variables for cma controller
        display = True
        generation = 0

        print("Starting CMA training")
        print("Generation {}".format(generation+1))

        while not es.stop(): 

            if cur_mean >= self.target_mean:
            # if cur_best <= self.stop_condiction:
                print("Already better than the target value")
                print("Stop training...")
                break

            # compute solutions
            result_list = [0] * self.pop_size  
            solutions = es.ask()

            if display: pbar = tqdm(total=self.pop_size*self.n_samples)
            
            for s_id, params in enumerate(solutions):
                for _ in range(self.n_samples):
                    value, _ = self.roller.rollout( 
                                                self, 
                                                params, 
                                                device=self.device,
                                                continuous=self.continuous,
                                                max_reward=self.max_reward
                                            )
                    result_list[s_id] += value / self.n_samples
                    if display: pbar.update(1)

            if display: pbar.close()

            # cma step
            es.tell(solutions, result_list)
            es.disp()

            # evaluation and saving
            print("Evaluating...")
            best_params, best, cur_mean = self.evaluate(solutions, result_list, run=6)

            print("Current evaluation of the objactive function (J): {} \nNote: this value should decrease".format((best))) 
            print("Current mean reward: {}".format(cur_mean)) 

            if not cur_best or cur_best > best: 
                print("Previous best with value J = {}...".format((cur_best)))
                cur_best = best
                print("Saving new best with value J = {}...".format((cur_best)))
    
                # load parameters into controller
                unflat_best_params = self.roller.unflatten_parameters(best_params, self.c.parameters(), self.device)
                for p, p_0 in zip(self.c.parameters(), unflat_best_params):
                    p.data.copy_(p_0)

                # saving
                self.c.save(self.modules_dir, f_value=cur_best)
                self.save()

                print("Rendering...")
                self.evaluate(solutions, result_list, render=True, run=3)

            if cur_mean >= self.target_mean:
            #if cur_best <= self.stop_condiction:
                print("Terminating controller training with value {}...".format(-cur_best))
                break

            generation += 1
            print("Generation {}".format(generation+1))

        return
    
    def evaluate(self, solutions, results, render=False, run=6):
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        best_estimates = []
        reward_estimate = []

        p_list = []
        for s_id in range(run):
            p_list.append((s_id, best_guess))

        for _ in tqdm(range(run)):
            value, reward = self.roller.rollout(
                                            self, 
                                            best_guess, 
                                            device=self.device, 
                                            render=render, 
                                            continuous=self.continuous,
                                            max_reward=self.max_reward
                                        )    
            best_estimates.append(value)
            reward_estimate.append(reward)
        return best_guess, np.mean(best_estimates), np.mean(reward_estimate)
    
#############################################################################
    
    def save(self):
        print("Saving model")
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    
