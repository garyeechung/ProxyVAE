import os

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
from tqdm import tqdm

from src.mnist.datasets import get_mnist_dataloaders, get_merged_labels
from src.mnist.losses import CVAE_Loss


MERGE_GROUP = [
    [1],
    [0, 6],
    [4, 7, 9],
    [2, 3, 5, 8]
]


def train_model(model, train_loader, optimizer, loss_fn, merge_group, device,
                return_each_batch=True):
    model.train()
    batch_losses = []

    for x, y in train_loader:
        x = x.to(device)
        ym = get_merged_labels(y, merge_group).float().to(device)
        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, ym)
        loss = loss_fn(recon_x, x, mu, logvar)
        loss.backward()
        optimizer.step()
        batch_losses.append(loss.item())

    if return_each_batch:
        return batch_losses
    else:
        return [sum(batch_losses) / len(train_loader)]


def evaluate_model(model, valid_loader, loss_fn, merge_group, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            ym = get_merged_labels(y, merge_group).float().to(device)
            recon_x, mu, logvar = model(x, ym)
            loss = loss_fn(recon_x, x, mu, logvar)
            total_loss += loss.item()
    return total_loss / len(valid_loader)


def train_cvae(model: Module, data_dir: str, ckpt_dir: str,
               beta1: float, device: str, merge_group=MERGE_GROUP,
               batch_size: int = 500, epochs: int = 300,
               lr: float = 5e-4, return_each_batch=True,
               replace_existing_ckpt=False):

    ckpt_dir = os.path.join(ckpt_dir, "invarep", f"beta1_{beta1}")
    ckpt_path = os.path.join(ckpt_dir, "cvae.pth")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    elif not replace_existing_ckpt and os.path.exists(ckpt_path):
        print(f"Checkpoint already exists at {ckpt_path}. Skipping training.")
        model.load_state_dict(torch.load(ckpt_path))
        print("Model loaded from checkpoint.")
        return model

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataloaders = get_mnist_dataloaders(data_dir, one_hot=False, batch_size=batch_size)
    train_loader, valid_loader, test_loader = dataloaders

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CVAE_Loss(beta=beta1)

    best_valid_loss = evaluate_model(model=model, valid_loader=valid_loader,
                                     loss_fn=loss_fn, merge_group=merge_group,
                                     device=device)

    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(epochs)):
        train_loss = train_model(model=model, train_loader=train_loader,
                                 optimizer=optimizer, loss_fn=loss_fn,
                                 merge_group=merge_group, device=device,
                                 return_each_batch=return_each_batch)
        train_losses = train_losses + train_loss

        valid_loss = evaluate_model(model=model, valid_loader=valid_loader,
                                    loss_fn=loss_fn, merge_group=merge_group,
                                    device=device)

        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)

    nb_minibatch = len(train_loader)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(range(nb_minibatch, nb_minibatch * epochs + 1, nb_minibatch),
             valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Minibatches')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(ckpt_dir, "cvae_losses.png"))
    plt.close()

    return model
