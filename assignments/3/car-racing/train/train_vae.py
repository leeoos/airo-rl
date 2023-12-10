import argparse
from os import mkdir, unlink, listdir, getpid, remove
from os.path import join, exists, dirname, abspath
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(dirname(dirname(abspath(__file__))))

from modules.vae import VAE 
from modules.vae import LATENT, OBS_SIZE

def train_vae(model, 
              data, 
              batch_size=32, 
              epochs=100, 
              lr=0.001, 
              device='cpu', 
              save='../checkpoints/'
    ):

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((OBS_SIZE, OBS_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


    transformed_data = torch.stack([transform_train(image) for image in data])

    dataset = TensorDataset(transformed_data, transformed_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    train_loss = 0
    log_step = (len(dataloader.dataset) // batch_size) // 2
    print("Training VAE")

    for epoch in range(epochs):
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
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % log_step == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch+1, 
                    batch_idx * len(inputs), 
                    len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    loss.item() / len(data)
                ))
    # end of trainig
    # insert here code for testing

    if save: model.save(save)
    return model


def loss_function(recon_x, x, mu, logsigma):
    """ VAE loss function """
    CE = F.cross_entropy(recon_x, x)
    KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
    return CE + KLD

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--epochs', type=int, help="training epochs")
    parser.add_argument('--batch', type=int, help="batch size")
    args = parser.parse_args()

    if args.dir:
        data_dir = args.dir 
    else:
        print("Error: no data")
        exit(1)

    batch = 32 if not args.batch else args.batch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    observations = torch.load(data_dir+'observations.pt')
    vae_model = VAE(3, LATENT).to(device)

    train_vae(
        model=vae_model,
        data=observations,
        epochs=args.epochs,
        batch_size=args.batch,
        device=device
    )

    

