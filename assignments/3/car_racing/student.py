import torch
import torch.nn as nn
from torchvision import transforms

from os import mkdir, remove, unlink, listdir, getpid
from os.path import join, exists
from time import sleep
from tqdm import tqdm
import sys

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
        # self.starting_state = True
        # self.memory = None # rnn state is empy at the beginning
        self.modules_dir = './checkpoints/'
       
        # models
        self.vae = VAE(img_channels=3, latent_size=LATENT).to(self.device)
        
        # self.rnn = MDN_RNN(
        #     input_size = self.latent_dim + self.action_space_dim, 
        #     output_size = self.latent_dim + self.action_space_dim
        # ).to(self.device)

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
        
        # c_in, z, h = self.forward(state)
        c_in = self.forward(state)
        a = self.c(c_in).to(self.device)

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

            data_dir = './dataset/'
            if not exists(data_dir):
                print("Error: no data")
                exit(1)

            observations = torch.load(data_dir+'observations.pt')

            self.vae = vae_trainer.train_vae(
                model=self.vae,
                data=observations,
                epochs=10,
                device=self.device,
                save='./checkpoints/'
            )
        else:
            self.vae = self.load_module(VAE(img_channels=3, latent_size=LATENT), self.modules_dir).to(self.device)
        
        if train_rnn: ...
        else: ...

        #################### TRAIN CONTROLLER  ###################################
        ##########################################################################
        train_controller = True # set to True to tarin controller

        if train_controller:
            pop_size = 4
            n_samples = 4 # 1 for signle thread

            params = self.c.parameters()
            flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
            es = cma.CMAEvolutionStrategy(flat_params, 0.1, {'popsize':pop_size})


            # define current best and load parameters
            cur_best = None
            ctrl_file = self.modules_dir+'controller.pt'
            print("Attempting to load previous best...")
            if exists(ctrl_file):
                state = torch.load(ctrl_file, map_location=self.device)
                cur_best = - state['reward']
                self.c.load_state_dict(state['state_dict'])
                print("Previous best was {}...".format(-cur_best))

            generation = 0
            log_step = 3 # print each n steps
            display = True
            cur_best = None
            target_return = 50 #950
            
            print("Wating for threds to start... ")
            print("Starting CMA training")

            while not es.stop(): # and generation < 20:

                if cur_best is not None and - cur_best > target_return:
                    print("Already better than target, breaking...")
                    break

                # Computing solutions
                r_list = [0] * pop_size  # result list
                solutions = es.ask()

                if display:
                    pbar = tqdm(total=pop_size * n_samples)
              
                for s_id, params in enumerate(solutions):
                    for _ in range(n_samples):
                        r_list[s_id] += self.roller.rollout(self.vae, self.c, params, device=self.device) / n_samples

                        if display:
                            pbar.update(1)

                if display:
                    pbar.close()

                es.tell(solutions, r_list)
                es.disp()

                # evaluation and saving
                if  generation % log_step == log_step - 1:

                    best_params, best = self.evaluate2(solutions, r_list)
                    print("Current evaluation: {}".format(-best))

                    if not cur_best or cur_best > best:
                        cur_best = best
                        std_best = 0 # tmp
                        print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
            
                        # load parameters into controller
                        for p, p_0 in zip(self.c.parameters(), best_params):
                            p.data.copy_(p_0)

                        torch.save(
                            {
                                'epoch': generation,
                                'reward': - cur_best,
                                'state_dict': self.c.state_dict()
                            },
                            ctrl_file
                        )
                    if - best > target_return:
                        print("Terminating controller training with value {}...".format(best))
                        break
                generation += 1
                print("End of generation: ", generation)

            
        return
    
    def evaluate2(self, solutions, results):
        print("Evaluating...")
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        value = self.roller.rollout(self.vae, self.c, best_guess, device=self.device)

        return best_guess, value
    
    #######################################################################################
    def load_module(self, model, model_dir):
        if exists(model_dir+model.name.lower()+'.pt'):
            print("Loading model "+model.name+" state parameters")
            model.load(model_dir)
            return model
        else:
            print("Error no model "+model.name.lower()+" found!")
            exit(1)
    
    def save(self):
        print("Saving model")
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret

    
