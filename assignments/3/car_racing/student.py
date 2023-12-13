import torch
import torch.nn as nn
from torchvision import transforms

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

from modules.vae import LATENT, OBS_SIZE
from modules.vae import VAE 

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
        self.n_samples = 2 
        self.temperature = 1000
        self.target_return = 950 + self.temperature

  
    def act(self, state):

        # convert input state to a torch tensor
        state = torch.tensor(state/255, dtype=torch.float)
        state = state.permute(0,2,1).permute(1,0,2)
        state = state.unsqueeze(0).to(self.device)

        # obs compression
        mu, logvar = self.vae.encode(state.float())
        z = self.vae.latent(mu, logvar)
        
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

            ####DEGUB####
            self.vae = self.vae.load('./foo/').to(self.device)
            # # self.vae = torch.load('./foo/vae.pt', map_location=torch.device(self.device))
            # X = torch.load(self.data_dir+'observations.pt').to(self.device)
            # samples = X[(np.random.rand(10)*X.shape[0]).astype(int)]
            # decodedSamples, _, _ = self.vae.forward(samples)
            
            # for index, obs in enumerate(samples):
            #     plt.subplot(5, 4, 2*index +1)
            #     obs = torch.movedim(obs, (1, 2, 0), (0, 1, 2)).cpu()
            #     plt.imshow(obs.numpy(), interpolation='nearest')

            # for index, dec in enumerate(decodedSamples):
            #     plt.subplot(5, 4, 2*index +2)
            #     decoded = torch.movedim(dec, (1, 2, 0), (0, 1, 2)).cpu()
            #     plt.imshow(decoded.detach().numpy(), interpolation="nearest")

            # plt.show()
            # sleep(2.)
            # plt.close()
            ####DEGUB####

############################ TRAIN CONTROLLER ###############################
#############################################################################

        # log variables for cma controller
        log_step = 3 # print log each n steps
        display = True
        generation = 0
        
        print("Attempting to load previous best...")
       
        # define current best as max
        cur_best = 100000000000 # max cap

        file_name = 'controller.pt' if not self.continuous else 'continuous.pt'
        if exists(self.modules_dir+file_name) and exists(self.modules_dir+file_name):
            with open(self.modules_dir+'cur_best.bk', 'r') as f : 
                for i in f : cur_best = float(i) 

            # load previous model
            print("Previous best was {}...".format(-cur_best))
            self.c = self.c.load(self.modules_dir)

        params = self.c.parameters()
        flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
        es = cma.CMAEvolutionStrategy(flat_params, 0.2, {'popsize':self.pop_size})

        print("Starting CMA training")
        print("Generation {}".format(generation+1))

        while not es.stop(): # and generation < 20:

            if cur_best is not None and (-cur_best) > self.target_return:
                print("Already better than target, breaking...")
                break

            # compute solutions
            r_list = [0] * self.pop_size  # result list
            solutions = es.ask()

            if display: pbar = tqdm(total=self.pop_size*self.n_samples)
            
            for s_id, params in enumerate(solutions):
                for _ in range(self.n_samples):
                    value = self.roller.rollout(self, 
                                                params, 
                                                device=self.device,
                                                continuous=self.continuous,
                                                temperature=self.temperature
                                        )
                    r_list[s_id] += value / self.n_samples
                    if display: pbar.update(1)

            if display: pbar.close()

            # cma step
            es.tell(solutions, r_list)
            es.disp()

            # evaluation and saving
            print("Evaluating...")
            best_params, best = self.evaluate(solutions, r_list)
            print("Current evaluation: {}".format(- best)) 

            if not cur_best or cur_best > best: 
                cur_best = best

                print("Saving new best with value {}...".format(-cur_best))
    
                # load parameters into controller
                for p, p_0 in zip(self.c.parameters(), best_params):
                    p.data.copy_(p_0)

                # saving
                self.c.save(self.modules_dir)
                with open(self.modules_dir+'cur_best.bk', 'w') as f: 
                    dump = cur_best 
                    f.write(str(dump))
                self.save()

                print("Rendering...")
                self.evaluate(solutions, r_list, render=True, run=3)

            if - best > self.target_return: #target_return:
                print("Terminating controller training with value {}...".format(-cur_best))
                break

            generation += 1
            print("Generation {}".format(generation+1))

        return
    
    
    def evaluate(self, solutions, results, render=False, run=6):
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        p_list = []
        for s_id in range(run):
            p_list.append((s_id, best_guess))

        for _ in tqdm(range(run)):
            value = self.roller.rollout(self, 
                                        best_guess, 
                                        device=self.device, 
                                        render=render, 
                                        continuous=self.continuous,
                                        temperature=self.temperature
                                )    
            restimates.append(value)
        return best_guess, np.mean(restimates)
    
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

    
