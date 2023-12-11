import gymnasium as gym

from os import mkdir, unlink, listdir, getpid, remove
from os.path import join, exists
import argparse
import pickle

import torch

import numpy as np
import math


def sample_continuous_actions(action_space, seq_len, dt):
        actions = [action_space.sample()]
        for _ in range(seq_len):
            daction_dt = np.random.randn(*actions[-1].shape)
            actions.append(
                np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                        action_space.low, action_space.high))
        return actions

def random_rollout(rollouts, data_dir): 
    env = gym.make('CarRacing-v2', continuous=True, render_mode='human')
    seq_len = 1500

    rollout_obs = []
    rollout_actions = []

    for i in range(rollouts):
        env.reset()
        action_list = sample_continuous_actions(env.action_space, seq_len, 1. / 50)

        _, _ = env.reset()
        done = False
        t = 0

        while not done and t < seq_len:
            action = action_list[t]
            t += 1
            observation, reward, terminated, truncated, info = env.step(action)

            observation = torch.from_numpy(observation).float() / 255
            rollout_obs.append(observation)

            action = torch.from_numpy(action).float()
            rollout_actions.append(action)

            done = terminated #or truncated

            if done or t == seq_len:
                print("> End of rollout {}, {} frames...".format(i, len(rollout_obs)))
                break

    # convert everyhing to a single torch tensor
    rollout_obs = torch.stack(rollout_obs, dim=0)
    rollout_obs = rollout_obs.permute(0,1,3,2).permute(0,2,1,3)
    rollout_actions = torch.stack(rollout_actions, dim=0)
    
    # save data on a file
    torch.save(rollout_obs, data_dir+'observations.pt')
    torch.save(rollout_actions, data_dir+'actions.pt')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    args = parser.parse_args()

    rollouts = args.rollouts if args.rollouts else 1

    data_dir = ''
    if args.dir:
        data_dir = args.dir 
    else:
        print("Error: no destination provided")
        exit(1)

    if not exists(data_dir): 
        mkdir(data_dir)
    else:
        if exists(data_dir+'observations.pt') : remove(data_dir+'observations.pt')
        if exists(data_dir+'actions.pt') : remove(data_dir+'actions.pt')

    # execute random rollouts to collect datas
    random_rollout(rollouts, data_dir)
