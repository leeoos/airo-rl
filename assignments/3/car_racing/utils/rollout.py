import gymnasium as gym
import torch
import torchvision.transforms as transforms

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

            observation = torch.from_numpy(observation).float() / 255
            rollout_obs.append(observation)

            action = torch.from_numpy(action).float()
            rollout_actions.append(action)
        
        rollout_obs = torch.stack(rollout_obs, dim=0)
        rollout_obs = rollout_obs.permute(0,1,3,2).permute(0,2,1,3)
        rollout_actions = torch.stack(rollout_actions, dim=0)
        return rollout_obs, rollout_actions
    

    def rollout(self, agent, controller, params=None, limit=100000):
        """ Execute a rollout and returns minus cumulative reward. """

        # copy params into the controller
        if params is not None:
            # params = self.unflatten_parameters(params, self.C.parameters(), self.device)

            # load parameters
            for p, p_0 in zip(controller.parameters(), params):
                p.data.copy_(p_0)

        obs, _ = self.env.reset()
        cumulative = 0
        i = 0

        while True:
            obs= torch.tensor(obs/255, dtype=torch.float).unsqueeze(0).permute(0,1,3,2).permute(0,2,1,3)
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            cumulative += reward
            if done or i > limit: return - cumulative
            i += 1
    