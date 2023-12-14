import gymnasium as gym

import sys
import argparse
from os import mkdir, unlink, listdir, getpid, remove
from os.path import join, exists, dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(42)

import numpy as np
import matplotlib.pyplot as plt

from modules.vae import VAE 
from modules.vae import LATENT, OBS_SIZE

def train_vae(model, 
              data, 
              batch_size_=32, 
              epochs=100, 
              lr_=0.001, 
              device='cpu', 
              save_dir='../checkpoints/'
    ):

    dataset = TensorDataset(data, data)
    dataloader = DataLoader(dataset, batch_size=batch_size_, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    model.train()
    training_losses = []  # List to store the training losses

    print("Training VAE")

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx,  data in enumerate(dataloader):
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward pass
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = loss_function(recon_batch, targets, mu, logvar)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            # Accumulate the loss
            epoch_loss += loss.item()

        # Calculate the average loss for the epoch
        average_loss = epoch_loss / len(dataloader)
        training_losses.append(average_loss)  # Store the average loss

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    if save_dir: 
        print("Saving model ...")
        model.save(save_dir)
        training_losses = np.array(training_losses)
        np.save('losses.npy', training_losses) 
    return model


def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
    BCE = bce_loss(recon_x, x)  
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return MSE + KLD
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--epochs', type=int, help="training epochs")
    parser.add_argument('--batch', type=int, help="batch size")
    args = parser.parse_args()

    data_dir = '../dataset/' if not args.dir else args.dir
    epochs = 10 if not args.epochs else args.epochs
    batch = 32 if not args.batch else args.batch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    observations = torch.load(data_dir+'observations.pt')
    vae_model = VAE().to(device)

    print("Dataset shape: {}".format(observations.shape))

    if args.train:
        vae_model = train_vae(
            model=vae_model,
            data=observations,
            epochs=epochs,
            batch_size_=batch,
            device=device,
            lr_=2.5e-4
        ).to(device)

    enable_test = True

    if args.test:

        if not args.train:
            modules_dir = '../checkpoints/'
            vae_model = vae_model.load(modules_dir).to(device)


        X = torch.load(data_dir+'observations.pt')
        # X = torch.stack(X, dim=0)
        # X = X.permute(0,1,3,2).permute(0,2,1,3).to(device)
        
        samples = X[(np.random.rand(10)*X.shape[0]).astype(int)]
        decoded_samples, _, _ = vae_model.forward(samples)
        
        for index, obs in enumerate(samples):
            plt.subplot(5, 4, 2*index +1)
            obs = torch.movedim(obs, (1, 2, 0), (0, 1, 2)).cpu()
            plt.imshow(obs.numpy(), interpolation='nearest')

        for index, dec in enumerate(decoded_samples):
            plt.subplot(5, 4, 2*index +2)
            decoded = torch.movedim(dec, (1, 2, 0), (0, 1, 2)).cpu()
            plt.imshow(decoded.detach().numpy(), interpolation="nearest")

        plt.show()



    
