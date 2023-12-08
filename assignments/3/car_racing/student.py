import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torch import Tensor, List
from torch.utils.data import Dataset
from torch.distributions.normal import Normal

import gymnasium as gym
import numpy as np
import os 

from utils.rollout import Rollout
from utils.module_trainer import Trainer
from modules.vae import VAE
from modules.mdn_rnn import MDN_RNN
import cma

class Policy(nn.Module):
    continuous = True # you can change this

    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(Policy, self).__init__()

        # gym env
        self.env = gym.make('CarRacing-v2', continuous=self.continuous, render_mode='robot')
        self.env.reset()

        # nn variables
        self.latent_dim = 100
        self.hidden_dim = 256
        self.action_space_dim = 3
        
        # global variables
        self.device = device
        self.batch_size = 32
        self.num_rollout = 1
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

        # random rollout to collect observations
        rollout_obs, rollout_actions = self.roller.random_rollout(self.env, self.num_rollout)
        rollout_obs.to(self.device)
        rollout_actions.detach().to(self.device)

        # train the vae
        self.vae = self.trainer.train(
            model_ =self.vae, 
            data_=rollout_obs, 
            batch_size_=self.batch_size,
            retrain=True
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
            retrain=True
        )

        print(self.roller.rollout(env=self.env, agent=self, controller=self.c))

        # train the controller
        # pop_size = 4
        # target_return= 950
        # params = self.C.parameters()
        # flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
        # print('test ', Rollout(self.env).rollout(self, self.c, params=flat_params))
        # es = cma.CMAEvolutionStrategy(flat_params, 0.1, {'popsize': pop_size})
        '''
        cur_best = None
        epoch = 0
        log_step = 3
        n_samples = 4
        display = True
        while not es.stop():
            
            if cur_best is not None and - cur_best > target_return:
                print("Already better than target, breaking...")
                break


            r_list = [0] * pop_size  # result list
            

            solutions = es.ask()


            # push parameters to queue
            for s_id, s in enumerate(solutions):
                for _ in range(n_samples):
                    self.p_queue.put((s_id, s))

            for _ in range(self.p_queue.qsize()):
                s_id, params = self.p_queue.get()
                self.r_queue.put((s_id, self.objective_function(params)))
                
          

            # retrieve results
            if display:
                pbar = tqdm(total=pop_size * n_samples)

            for _ in range(pop_size * n_samples):
                while self.r_queue.empty():
                    sleep(.1)

                r_s_id, r = self.r_queue.get()
                r_list[r_s_id] += r / n_samples

                if display:
                    pbar.update(1)

            if display:
                pbar.close()

            es.tell(solutions, r_list)
            es.disp()


            # evaluation and saving
            if epoch % log_step == log_step - 1:
                best_params, best, std_best = self.evaluate(solutions, r_list, self.p_queue, self.r_queue)
                print("Current evaluation: {}".format(best))

                if not cur_best or cur_best > best:
                    cur_best = best
                    print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                    
                    # load_parameters(best_params, controller)

                    # Load parameters
                    for p, p_0 in zip(self.C.parameters(), params):
                        p.data.copy_(p_0)
                    controller = self.C
                    # torch.save(
                    #     {'epoch': epoch,
                    #     'reward': - cur_best,
                    #     'state_dict': controller.state_dict()},
                    #     join(ctrl_dir, 'best.tar'))
                if - best > target_return:
                    print("Terminating controller training with value {}...".format(best))
                    break


            epoch += 1

        # es.result_pretty()
        self.e_queue.put('EOP')
        '''

    def save(self):
        torch.save(self.state_dict(), 'model.pt')

    def load(self): 
        self.load_state_dict(torch.load('model.pt', map_location=self.device))

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        return ret
