import cma
import torch
import torch.nn as nn
import numpy as np

class CMA():
    def __init__(self, controller) -> None:
        self.controller = controller
        pass

    def objective_function(self, params):
        # Reshape params and set them to the layer

        with torch.no_grad():
            weights_shape = self.controller.weight.shape
            bias_shape = self.controller.bias.shape
            weights_flat_size = np.prod(weights_shape)
            weights = params[:weights_flat_size].reshape(weights_shape)
            biases = params[weights_flat_size:].reshape(bias_shape)
            self.controller.weight.copy_(torch.tensor(weights))
            self.controller.bias.copy_(torch.tensor(biases))

        # Compute the loss using the controller here
        # loss = 

        return loss.item()
    
    def train(self):

        # Flatten initial parameters
        initial_weights = self.controller.weight.flatten().detach().numpy()
        initial_biases = self.controller.bias.flatten().detach().numpy()
        initial_params = np.concatenate([initial_weights, initial_biases])

        # Initialize CMA-ES
        options = {'maxiter': 1000, 'tolx': 1e-4, 'popsize': 10}
        es = cma.CMAEvolutionStrategy(initial_params, 0.5, options)

        # Optimization loop
        while not es.stop():
            solutions = es.ask()
            es.tell(solutions, [objective_function(s) for s in solutions])
            es.disp()

        # Set optimized parameters
        best_params = es.result.xbest
        controller.weight.copy_(torch.tensor(best_params[:weights_flat_size].reshape(weights_shape)))
        controller.bias.copy_(torch.tensor(best_params[weights_flat_size:].reshape(bias_shape)))
