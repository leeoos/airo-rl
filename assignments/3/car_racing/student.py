import gymnasium as gym
import cma

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from torchvision import transforms
from torch import Tensor, List
from torch.multiprocessing import Process, Queue
from torch.distributions.normal import Normal

from os import mkdir, remove, unlink, listdir, getpid
from os.path import join, exists
from tqdm import tqdm
from time import sleep
import numpy as np
import sys

from data.rollout import Rollout
from modules.vae import VAE 
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

        # multiprocess
        self.p_queue = Queue()
        self.r_queue = Queue()
        self.e_queue = Queue()
        
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
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((OBS_SIZE, OBS_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.roller = Rollout()


    def forward(self, x):

        # if self.starting_state:
        #     h = torch.zeros(1, self.hidden_dim).to(self.device)
        #     self.starting_state = False
        # else: h = self.memory

        mu, logvar = self.vae.encode(x.float())
        z = self.vae.latent(mu, logvar)

        # c_in = torch.cat((z, h),dim=1).to(self.device)    
        return z #c_in, z, h
    
    def act(self, state):
        # convert input state to a torch tensor
        state = torch.tensor(state/255, dtype=torch.float)
        state = self.transform(state.permute(0,2,1).permute(1,0,2))
        state = state.unsqueeze(0).to(self.device)
        # state = state.unsqueeze(0).permute(0,1,3,2).permute(0,2,1,3).to(self.device)

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

        # set to True to retarin vae and rnn
        train_vae = False
        train_rnn = False

        if train_vae: 

            data_dir = './data/dataset/'
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

        #################### TRAIN CONTROLLER  MULTITHRED ########################
        ##########################################################################
        train_controller = True # set to True to retarin controller

        if train_controller:
            pop_size = 6
            n_samples = 6 # 1 for signle thread
            num_workers = 8

            ### START THREDS ###

            tmp_dir = 'log/'
            if not exists(tmp_dir):
                mkdir(tmp_dir)
            else:
                # remove(tmp_dir)
                # mkdir(tmp_dir)
                for fname in listdir(tmp_dir):
                    unlink(join(tmp_dir, fname))

            list_of_process = []
            for p_index in range(num_workers):
                p = Process(
                    target=self.slave_routine, 
                    args=(self.p_queue, self.r_queue, self.e_queue, p_index, tmp_dir)
                )
                p.start()
                list_of_process.append(p)
            
            ### END THREDS ###

            params = self.c.parameters()
            flat_params = torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()
            es = cma.CMAEvolutionStrategy(flat_params, 0.1, {'popsize':pop_size})

            # sleep(10.)
            # generations = 0
            # while not es.stop():
            #     print("Generation: ", generations)
            #     r_list = [0] * pop_size  # result list
            #     solutions = es.ask()
            #     for s_id, s in enumerate(solutions):
            #         for _ in range(n_samples):
            #             self.p_queue.put((s_id, s))
            #     print('Pqueue : ', self.p_queue.qsize())

            #     while self.r_queue.empty():
            #         sleep(.1)
            #     print('Rqueue : ', self.r_queue.qsize())
            #     sleep(2.)
            #     generations += 1
            #     if generations >= 10 : break
            

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

            while not es.stop() and generation < 20:

                if cur_best is not None and - cur_best > target_return:
                    print("Already better than target, breaking...")
                    break

                # Computing solutions
                r_list = [0] * pop_size  # result list
                solutions = es.ask()

                # push parameters to queue
                for s_id, s in enumerate(solutions):
                    for _ in range(n_samples):
                        self.p_queue.put((s_id, s))
                
                if display:
                    pbar = tqdm(total=pop_size * n_samples)

                for _ in range(pop_size * n_samples):

                    # ctrl = 0
                    while self.r_queue.empty():
                        sleep(1.)

                    # print('R queue : ', self.r_queue.qsize())
                    r_s_id, r = self.r_queue.get()
                    r_list[r_s_id] += r / n_samples

                    if display:
                        pbar.update(1)

                if display:
                    pbar.close()

                es.tell(solutions, r_list)
                # es.disp()

                # evaluation and saving
                if  generation % log_step == log_step - 1:

                    # print("copy of r_queue: ", c_queue.qsize())
                    # best_params, best = self.evaluate(solutions, r_list, p_queue=cp_queue, r_queue=c_queue)
                    best_params, best, std_best = self.evaluate(solutions, r_list, rollouts=24)
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

            self.e_queue.put('EOP')
            for p in list_of_process:
                p.terminate()
        return
    ##########################################################################
  
    def slave_routine(self, p_queue, r_queue, e_queue, p_index, tmp_dir):

        # redirect streams
        sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
        sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')
        
        with torch.no_grad():
            roller = Rollout()

            # with open('foo', 'a') as f: f.write('e: '+str(self.e_queue.qsize())+'\n')
            while self.e_queue.empty():
                if self.p_queue.empty():
                    ...
                else:
                    # with open('foo', 'a') as f: f.write('p: '+str(self.p_queue.qsize())+'\n')
                    s_id, params = self.p_queue.get()
                    # self.r_queue.put((s_id, self.roller.rollout(self.env, self, self.c, params))
                    value = roller.rollout(self.vae, self.c, params, device=self.device)
                    # value = self.roller.sasso()
                    # with open('foo', 'a') as f: f.write('v: '+str(value)+'\n')
                    # value = 42
                    self.r_queue.put((s_id, value))
                    # with open('foo', 'a') as f: f.write('r: '+str(self.r_queue.qsize())+'\n')
            print("End of my life")


    def evaluate(self, solutions, results, rollouts=100):
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        for s_id in range(rollouts):
            self.p_queue.put((s_id, best_guess))

        print("Evaluating...")
        for _ in tqdm(range(rollouts)):
            while self.r_queue.empty():
                sleep(.1)
            restimates.append(self.r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)
    
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

    
