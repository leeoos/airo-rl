import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import copy

class Trainer():
    def __init__(self):
        pass

    def train_module(self, model, loss_fn, optimizer, data, batch_size, num_epochs):
        for epoch in range(num_epochs):
            for i in range(0, len(data), batch_size):
                X_batch = data[i:i+batch_size]
                y_batch = data[i:i+batch_size]

                out, mu, logvar, sigma, pi = model.forward(X_batch)
                loss = loss_fn(out, y_batch, mu, logvar, sigma, pi)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Print the loss every 100 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Model_{model.name}: Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        return model
    
    def train(self, model_, data_, batch_size_=32, epochs_=100, lr_=0.001):
      
        trained_model = copy.deepcopy(model_)

        print("Training " + model_.name.lower())
        trained_model = self.train_module(
            model=model_, 
            loss_fn=model_.loss,
            optimizer=torch.optim.Adam(model_.parameters(), lr=lr_), 
            data=data_, 
            batch_size=batch_size_, 
            num_epochs=epochs_
        )
        trained_model.save()

        return trained_model
    
    def load_model(self, model_):
        if os.path.exists('./models/'+model_.name.lower()+'.pt'):
            print("Loading model "+model_.name+" state parameters")
            model_.load()
            return model_
        else:
            print("Error no model "+model_.name.lower()+" found!")
            return None
    
    
        
