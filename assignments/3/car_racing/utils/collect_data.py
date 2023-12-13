import gymnasium as gym

import pickle
import argparse
from os.path import join, exists
from os import mkdir, unlink, listdir, getpid, remove

import torch
torch.manual_seed(42)


import math
import numpy as np

# to check
def sample_continuous_actions(action_space, seq_len, dt):
        actions = [action_space.sample()]
        for _ in range(seq_len):
            daction_dt = np.random.randn(*actions[-1].shape)
            actions.append(
                np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                        action_space.low, action_space.high)
            )
        return actions

def random_rollout(rollouts, continuous, render, data_dir): 

    render_mode = 'human' if render else 'rgb_array'
    env = gym.make('CarRacing-v2', continuous=continuous, render_mode=render_mode)
    seq_len = 1200

    rollout_obs = []
    rollout_actions = []

    for i in range(rollouts):
        env.reset()
        action_list = []

        if continuous: 
            action_list = sample_continuous_actions(env.action_space, seq_len, 1. / 50)

        _, _ = env.reset()
        action = None
        done = False
        t = 0

        while not done and t < seq_len:

            if continuous: 
                action = action_list[t]
                save_action = torch.from_numpy(action).float()
                rollout_actions.append(save_action)

            else: 
                # select a random action except do nothing 
                action = int(np.random.rand()*3 + 1) 
                
            t += 1
            observation, reward, terminated, truncated, info = env.step(action)
            observation = torch.from_numpy(observation).float() / 255
            rollout_obs.append(observation)

            done = terminated #or truncated

            if done or t == seq_len:
                print("> End of rollout {}, {} frames...".format(i+1, len(rollout_obs)))
                break

    # convert observatons to a single torch tensor and save on pt file
    rollout_obs = torch.stack(rollout_obs, dim=0)
    rollout_obs = rollout_obs.permute(0,1,3,2).permute(0,2,1,3)
    # torch.save(rollout_obs, data_dir+'observations.pt')

    # same for actions if continuous
    if continuous:
        rollout_actions = torch.stack(rollout_actions, dim=0)
        # torch.save(rollout_actions, data_dir+'actions.pt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--render', action='store_true')
    parser.add_argument('-c', '--continuous', action='store_true')
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    args = parser.parse_args()

    rollouts = args.rollouts if args.rollouts else 1
    data_dir = args.dir if args.dir else '../dataset/'

    if not exists(data_dir): 
        mkdir(data_dir)
    else:
        if exists(data_dir+'observations.pt') : remove(data_dir+'observations.pt')
        if exists(data_dir+'actions.pt') : remove(data_dir+'actions.pt')

    # execute random rollouts to collect datas
    random_rollout(rollouts, args.continuous, args.render, data_dir)
