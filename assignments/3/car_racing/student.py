import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor, List
from torch.utils.data import Dataset
# from torch.multiprocessing import Process, Queue
from torch.distributions.normal import Normal

from queue import Queue
from tqdm import tqdm
from time import sleep
from copy import copy
import gymnasium as gym
import matplotlib as plt
import numpy as np
import os 
import cma

from utils.rollout import Rollout
from utils.module_trainer import Trainer
from modules.vae import VAE
from modules.mdn_rnn import MDN_RNN

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()

        # gym env
        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='human')

        # nn variables
        self.latent_dim = 100
        self.hidden_dim = 256
        self.action_space_dim = 3
        
        # global variables
        self.device = device
        self.batch_size = 32
        self.num_rollout = 32*100
        self.starting_state = True
        self.memory = None # rnn state is empy at the beginning
       
        # models
        self.vae = VAE(latent_size=self.latent_dim).to(self.device)
        self.rnn = MDN_RNN(
            input_size = self.latent_dim + self.action_space_dim, 
            output_size = self.latent_dim + self.action_space_dim
        ).to(self.device)
        self.c = nn.Linear(
            in_features= self.latent_dim + self.hidden_dim, 
            out_features= self.action_space_dim
        ).to(self.device)

        # utils
        self.trainer = Trainer()
        self.roller = Rollout()


    def forward(self, x):

        if self.starting_state:
            h = torch.zeros(1, self.hidden_dim).to(self.device)
            self.starting_state = False

        else: h = self.memory
        # print("h: ", h.shape)

        mu, logvar = self.vae.encode(x.float())
        z = self.vae.latent(mu, logvar)
        # print("z: ", z.shape)

        c_in = torch.cat((z, h),dim=1).to(self.device)
        # print("c_in: ", c_in.shape)       
        return c_in, z, h
    
    def act(self, state):
        c_in, z, h = self.forward(state)
        a = self.c(c_in).to(self.device)
        # print("action: ", a.shape)

        rnn_in = torch.concat((a, z), dim=1).to(self.device)
        # print("rnn_in: ", rnn_in.shape)

        self.memory, _ = self.rnn.forward_lstm(rnn_in)
        self.memory = self.memory.squeeze(0).to(self.device)
        # print("memory: ", self.memory.shape)
        
        torch.clip(a, min = -1, max = 1 )
        return a.cpu().float().squeeze().detach().numpy()

    def train(self):
        """Train the entire network or just the controller module"""

        train_nn_modules = True # set to True to retarin vae and rnn

        if train_nn_modules:

            # random rollout to collect observations
            rollout_obs, rollout_actions = self.roller.random_rollout(self.env, self.num_rollout)
            # print(rollout_obs.shape)
            rollout_obs.to(self.device)
            rollout_actions.detach().to(self.device)

            # train the vae
            self.vae = self.trainer.train(
                model_ =self.vae, 
                data_=rollout_obs, 
                batch_size_=self.batch_size,
                epochs_=200
            )

            # encode the observation to train the rnn
            mu, logvar = self.vae.encode(rollout_obs)
            rollout_latent = self.vae.latent(mu, logvar).to(self.device)
            rollout_al = torch.cat((rollout_actions, rollout_latent), dim=1).to(self.device)

            # print(self.rnn.forward_lstm(rollout_al)[0].shape)

            # train the rnn
            self.rnn = self.trainer.train(
                model_ =self.rnn, 
                data_=rollout_al.detach(), 
                batch_size_=self.batch_size,
                epochs_=50
            )

        # print(self.roller.rollout(env=self.env, agent=self, controller=self.c))

        ##########################################################################
        train_controller = False # set to True to retarin controller

        if train_controller:
            pop_size = 4
            n_samples = 1 # single
            target_return= 950

            params = self.c.parameters()
            flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

            #print('test ', self.roller.rollout(self.env, self, self.c, params=flat_params))
            es = cma.CMAEvolutionStrategy(flat_params, 0.1, {'popsize':pop_size})

            p_queue = Queue()
            r_queue = Queue()
            e_queue = Queue()
            c_queue = Queue()

            epoch = 0
            log_step = 3
            display = True
            cur_best = None
            target_return = 50 #950

            print("Starting CMA training")
            while not es.stop():
                if cur_best is not None and - cur_best > target_return:
                    print("Already better than target, breaking...")
                    break

                r_list = [0] * pop_size  # result list
                solutions = es.ask()

                # print('r_list len: ', len(r_list))
                # print('solution len: ', len(solutions))
                # print('solution shape: ', solutions[0].shape)
                # print('parameters shape:', flat_params.shape)

                # push parameters to queue
                for s_id, s in enumerate(solutions):
                    for _ in range(n_samples):
                        p_queue.put((s_id, s))
                
                # print('queue: ', p_queue.qsize())
                # print('range: ', pop_size*n_samples)

                for _ in range(pop_size * n_samples):
                    s_id, params = p_queue.get()
                    r_queue.put((
                            s_id, 
                            self.roller.rollout(self.env, self, self.c, params, display=True)
                        ))

                # retrieve results
                if display:
                    pbar = tqdm(total=pop_size * n_samples)

                for _ in range(pop_size * n_samples):

                    r_s_id, r = r_queue.get(p_queue.qsize())
                    r_list[r_s_id] += r / n_samples
                    c_queue.put((r_s_id, r))

                    if display:
                        pbar.update(1)

                if display:
                    pbar.close()

                # print('queue: ', p_queue.qsize())
                es.tell(solutions, r_list)
                es.disp()

                # evaluation and saving
                if epoch % log_step == log_step - 1:

                    print("copy of r_queue: ", c_queue.qsize())
                    best_params, best, std_best = self.roller.evaluate(solutions, r_list, r_queue=c_queue)
                    # best_params, best, std_best = self.roller.evaluate(solutions, r_list, r_queue=r_queue)
                    print("Current evaluation: {}".format(best))

                    if not cur_best or cur_best > best:
                        cur_best = best
                        print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
            
                        # load parameters into controller
                        for p, p_0 in zip(self.c.parameters(), params):
                            p.data.copy_(p_0)

                    if - best > target_return:
                        print("Terminating controller training with value {}...".format(best))
                        break

                epoch += 1

                # break
        return
        ##########################################################################

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
