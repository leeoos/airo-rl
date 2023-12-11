import torch
import torch.nn as nn
from torchvision import transforms

from os import mkdir, remove, unlink, listdir, getpid
from os.path import join, exists
from time import sleep
from tqdm import tqdm
import time
import sys

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import cma

from modules.vae import VAE 
from utils.rollout import Rollout
from modules.vae import LATENT, OBS_SIZE
import train.train_vae as vae_trainer
# from modules.mdn_rnn import MDN_RNN

ACTIONS = 3

class Policy(nn.Module):
    
    continuous = True # you can change this

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
        
        # self.rnn = MDN_RNN(
        #     input_size = self.latent_dim + self.action_space_dim, 
        #     output_size = self.latent_dim + self.action_space_dim
        # ).to(self.device)
        # self.starting_state = True
        # self.memory = None # rnn state is empy at the beginning

        self.c = nn.Linear(
            in_features= LATENT, #+ self.hidden_dim, 
            out_features= ACTIONS
        ).to(self.device)

        # utils
        self.roller = Rollout()

  
    def act(self, state):
        # convert input state to a torch tensor
        state = torch.tensor(state/255, dtype=torch.float)
        state = state.permute(0,2,1).permute(1,0,2)
        state = state.unsqueeze(0).to(self.device)

        # obs compression
        mu, logvar = self.vae.encode(state.float())
        z = self.vae.latent(mu, logvar)
        
        # c_in, z, h = ...
        a = self.c(z).to(self.device)

        # rnn_in = torch.concat((a, z), dim=1).to(self.device)
        # _, hidden = self.rnn.forward_lstm(rnn_in) # lstm outputs: out, (h_n, c_n)
        # self.memory = hidden[0].squeeze(0).to(self.device)
        
        torch.clip(a, min = -1, max = 1 )
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

            ####DEGUB####
            X = torch.load(self.data_dir+'observations.pt').to(self.device)
            samples = X[(np.random.rand(10)*X.shape[0]).astype(int)]
            decodedSamples, _, _ = self.vae.forward(samples)
            
            for index, obs in enumerate(samples):
                plt.subplot(5, 4, 2*index +1)
                obs = torch.movedim(obs, (1, 2, 0), (0, 1, 2)).cpu()
                plt.imshow(obs.numpy(), interpolation='nearest')

            for index, dec in enumerate(decodedSamples):
                plt.subplot(5, 4, 2*index +2)
                decoded = torch.movedim(dec, (1, 2, 0), (0, 1, 2)).cpu()
                plt.imshow(decoded.detach().numpy(), interpolation="nearest")

            plt.show()
            sleep(2.)
            plt.close()
            ####DEGUB####
        else:
            self.vae = self.vae.load(self.modules_dir).to(self.device)

            ####DEGUB####
            X = torch.load(self.data_dir+'observations.pt').to(self.device)
            samples = X[(np.random.rand(10)*X.shape[0]).astype(int)]
            decodedSamples, _, _ = self.vae.forward(samples)
            
            for index, obs in enumerate(samples):
                plt.subplot(5, 4, 2*index +1)
                obs = torch.movedim(obs, (1, 2, 0), (0, 1, 2)).cpu()
                plt.imshow(obs.numpy(), interpolation='nearest')

            for index, dec in enumerate(decodedSamples):
                plt.subplot(5, 4, 2*index +2)
                decoded = torch.movedim(dec, (1, 2, 0), (0, 1, 2)).cpu()
                plt.imshow(decoded.detach().numpy(), interpolation="nearest")

            plt.show()
            sleep(2.)
            plt.close()
            ####DEGUB####

        
        if train_rnn: ...
        else: ...

        ########################### TRAIN CONTROLLER  ############################
        ##########################################################################

        # ####DEGUB####
        # for p in self.c.parameters():
        #     print('previous parameters: {}'.format(p))
        #     break
        # ####DEGUB####

        # training parameters
        pop_size = 4
        n_samples = 4 
        generation = 0
        target_return = 200 #950

        # log variables
        log_step = 3 # print log each n steps
        display = True
        render = False
        
        # define current best and load parameters
        cur_best = None
        c_checkpoint = self.modules_dir+'controller.pt'
        print("Attempting to load previous best...")
        if exists(c_checkpoint):
            state = torch.load(c_checkpoint, map_location=self.device)
            cur_best = - state['reward']
            self.c.load_state_dict(state['state_dict'])
            print("Previous best was {}...".format(-cur_best))

        params = self.c.parameters()
        flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
        es = cma.CMAEvolutionStrategy(flat_params, 0.1, {'popsize':pop_size})

        print("Starting CMA training")
        start_time = time.time()

        while not es.stop(): # and generation < 20:

            if cur_best is not None and - cur_best > target_return:
                print("Already better than target, breaking...")
                break

            # compute solutions
            r_list = [0] * pop_size  # result list
            solutions = es.ask()

            if display: pbar = tqdm(total=pop_size * n_samples)
            
            for s_id, params in enumerate(solutions):
                for _ in range(n_samples):
                    r_list[s_id] += self.roller.rollout(self, params, device=self.device) / n_samples
                    if display: pbar.update(1)

            if display: pbar.close()

            es.tell(solutions, r_list)
            es.disp()

            # evaluation and saving
            if  generation % log_step == log_step - 1:

                # render = True
                best_params, best = self.evaluate(solutions, r_list, render)
                print("Current evaluation: {}".format(-best))

                if not cur_best or cur_best > best:
                    cur_best = best
                    print("Saving new best with value {}...".format(-cur_best))
        
                    # load parameters into controller
                    for p, p_0 in zip(self.c.parameters(), best_params):
                        p.data.copy_(p_0)

                    torch.save(
                        {
                            'epoch': generation,
                            'reward': - cur_best,
                            'state_dict': self.c.state_dict()
                        },
                        c_checkpoint
                    )

                print("--- %s seconds since start training ---" % (time.time() - start_time)) 

                if - cur_best > target_return:
                    print("Terminating controller training with value {}...".format(-cur_best))
                    break

            generation += 1
            render = False
            print("End of generation: ", generation)
            
        return
    
    
    def evaluate(self, solutions, results, render):
        print("Evaluating...")
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        value = self.roller.rollout(self, best_guess, device=self.device, render=render)
        return best_guess, value
    
    ##########################################################################
    
    def save(self):
        print("Saving model")
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    
