import gymnasium as gym
import torch
import torchvision.transforms as transforms

class Rollout():

    def __init__(self):
        pass

    def random_rollout(self, env,num_rollout=1):
        """ Execute a random rollout and returns actions and observations """

        rollout_obs = []
        rollout_actions = []

        for _ in range(num_rollout):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
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
    
    def unflatten_parameters(self, params, example, device):
        """ Unflatten parameters. NOte: example is generator of parameters (as returned by module.parameters()), used to reshape params """

        params = torch.Tensor(params).to(device)
        idx = 0
        unflattened = []
        for e_p in example:
            unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
            idx += e_p.numel()
        return unflattened

    def rollout(self, env, agent, controller, params=None, limit=100000, device='cpu'):
        """ Execute a rollout and returns minus cumulative reward. """

        if params is not None:
            params = self.unflatten_parameters(params, controller.parameters(), device)

            # load parameters into controller
            for p, p_0 in zip(controller.parameters(), params):
                p.data.copy_(p_0)

        obs, _ = env.reset()
        cumulative = 0
        i = 0

        while True:
            obs= torch.tensor(obs/255, dtype=torch.float).unsqueeze(0).permute(0,1,3,2).permute(0,2,1,3)
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            cumulative += reward
            if done or i >= limit: return - cumulative
            i += 1

    
    