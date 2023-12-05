import cma
import torch
import torch.nn as nn
import numpy as np

class CCMA(nn.Module):

    def __init__(self, env, input_dim, output_dim):
        super().__init__()

        self.name = 'CCMA'
        self.env = env
        self.c = nn.Linear(input_dim, output_dim)

    def objective_function(self, params):

        # Reshape params and set them to the layer
        with torch.no_grad():
            weights_shape = self.c.weight.shape
            bias_shape = self.c.bias.shape
            weights_flat_size = np.prod(weights_shape)
            weights = params[:weights_flat_size].reshape(weights_shape)
            biases = params[weights_flat_size:].reshape(bias_shape)
            self.c.weight.copy_(torch.tensor(weights))
            self.c.bias.copy_(torch.tensor(biases))

        # Compute the loss using the controller
        cumulative_reward = 0
        done = False
        state = self.env.reset()

        while not done:
            # Use the controller to determine the action
            action = self.c(state, params)
            state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            cumulative_reward += reward

        # Since CMA-ES minimizes, return negative reward
        return -cumulative_reward
    
    def train(self):

        # Flatten initial parameters
        initial_weights = self.c.weight.flatten().detach().numpy()
        initial_biases = self.c.bias.flatten().detach().numpy()
        initial_params = np.concatenate([initial_weights, initial_biases])

        # Initialize CMA-ES
        options = {'maxiter': 1000, 'tolx': 1e-4, 'popsize': 10}
        es = cma.CMAEvolutionStrategy(initial_params, 0.5, options)

        # Optimization loop
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [self.objective_function(s) for s in solutions])
            es.disp()

        # Set optimized parameters
        best_params = es.result.xbest
        self.c.weight.copy_(torch.tensor(best_params[:self.c.weights_flat_size].reshape(self.c.weights_shape)))
        self.c.bias.copy_(torch.tensor(best_params[self.c.weights_flat_size:].reshape(self.c.bias_shape)))




