import argparse
import random
import numpy as np
from student import Policy
import gymnasium as gym

import torch 
import matplotlib.pyplot as plt

def evaluate(env=None, n_episodes=1, render=False):
    agent = Policy()
    agent.load()

    go = 0 
    ####DEGUB####
    for p in agent.c.parameters():
        print('previous parameters: {}'.format(p))
        go += 1
        if go == 3 :break
    ####DEGUB####

    # X = torch.load(agent.data_dir+'observations.pt').to(agent.device)
    # samples = X[(np.random.rand(10)*X.shape[0]).astype(int)]
    # decodedSamples, _, _ = agent.vae.forward(samples)
    
    # for index, obs in enumerate(samples):
    #     plt.subplot(5, 4, 2*index +1)
    #     obs = torch.movedim(obs, (1, 2, 0), (0, 1, 2)).cpu()
    #     plt.imshow(obs.numpy(), interpolation='nearest')

    # for index, dec in enumerate(decodedSamples):
    #     plt.subplot(5, 4, 2*index +2)
    #     decoded = torch.movedim(dec, (1, 2, 0), (0, 1, 2)).cpu()
    #     plt.imshow(decoded.detach().numpy(), interpolation="nearest")

    # plt.show()
    # ####DEGUB####
    return

    env = gym.make('CarRacing-v2', continuous=agent.continuous)
    if render:
        env = gym.make('CarRacing-v2', continuous=agent.continuous, render_mode='human')
        
    rewards = []
    for episode in range(n_episodes):
        total_reward = 0
        done = False
        s, _ = env.reset()
        while not done:
            action = agent.act(s)
            
            s, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        rewards.append(total_reward)
        
    print('Mean Reward:', np.mean(rewards))


def train():
    agent = Policy()
    agent.train()
    print("I am here")
    agent.save()
    print("I am here 2")

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    args = parser.parse_args()

    if args.train:
        train()

    if args.evaluate:
        evaluate(render=args.render)

    
if __name__ == '__main__':
    main()
