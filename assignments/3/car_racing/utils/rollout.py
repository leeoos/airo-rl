import gymnasium as gym
import torch

class Rollout():

    def __init__(self, env) -> None:
        self.env = env 
        self.env.reset()

    def random_rollout(self, num_rollout=1):
        rollout_obs = []
        rollout_actions = []

        for _ in range(num_rollout):
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if done: break

            observation = torch.from_numpy(observation).float()
            rollout_obs.append(observation)

            action = torch.from_numpy(action).float()
            rollout_actions.append(action)
        
        rollout_obs = torch.stack(rollout_obs, dim=0)
        rollout_obs = rollout_obs.permute(0,1,3,2).permute(0,2,1,3)

        rollout_actions = torch.stack(rollout_actions, dim=0)

        return rollout_obs, rollout_actions