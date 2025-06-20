import os

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
import torch.optim as optim
from tqdm import tqdm

from src.mnist.datasets import get_mnist_dataloaders, get_merged_labels, MERGE_GROUP
from src.mnist.losses import InfoBottleneck_Loss


def train_model(model, train_loader, optimizer, loss_fn, merge_group, device,
                return_each_batch=True):
    model.train()
    batch_losses = []

    for x, y in train_loader:
        x = x.to(device)
        y = get_merged_labels(y, merge_group).float().to(device)
        optimizer.zero_grad()
        y_pred, mu, logvar = model(x)
        loss = loss_fn(y, y_pred, mu, logvar)
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
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            ym = get_merged_labels(y, merge_group).float().to(device)
            ym_pred, mu, logvar = model(x)
            loss = loss_fn(ym, ym_pred, mu, logvar)
            total_loss += loss.item()

            ym_pred = torch.argmax(ym_pred, dim=-1)
            ym = torch.argmax(ym, dim=-1)
            correct += (ym_pred == ym).sum().item()
            total += ym.size(0)
    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(valid_loader), accuracy


def train_infobottleneck_groupifier(model: Module, data_dir: str, ckpt_dir: str,
                                    beta: float, device: str, merge_group=MERGE_GROUP,
                                    batch_size: int = 500, epochs: int = 300,
                                    lr: float = 5e-4, return_each_batch=True,
                                    replace_existing_ckpt=False):

    ckpt_dir = os.path.join(ckpt_dir, "information_bottleneck", f"beta_{beta}")
    ckpt_path = os.path.join(ckpt_dir, "groupifier.pth")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    elif not replace_existing_ckpt and os.path.exists(ckpt_path):
        print(f"Checkpoint already exists at {ckpt_path}. Skipping training.")
        model.load_state_dict(torch.load(ckpt_path))
        print("Model loaded from checkpoint.")
        return

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dataloaders = get_mnist_dataloaders(data_dir, one_hot=False, batch_size=batch_size)
    train_loader, valid_loader, test_loader = dataloaders

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = InfoBottleneck_Loss(beta=beta)

    best_valid_loss, _ = evaluate_model(model=model, valid_loader=valid_loader,
                                        loss_fn=loss_fn, merge_group=merge_group,
                                        device=device)

    train_losses = []
    valid_losses = []
    valid_accuracies = []
    for epoch in tqdm(range(epochs)):
        train_loss = train_model(model=model, train_loader=train_loader,
                                 optimizer=optimizer, loss_fn=loss_fn,
                                 merge_group=merge_group, device=device,
                                 return_each_batch=return_each_batch)
        train_losses = train_losses + train_loss

        valid_loss, valid_acc = evaluate_model(model=model, valid_loader=valid_loader,
                                               loss_fn=loss_fn, merge_group=merge_group,
                                               device=device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)

    nb_minibatches = len(train_loader)
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    axes[0].plot(range(nb_minibatches, nb_minibatches * epochs + 1, nb_minibatches),
                 valid_losses, label='Validation Loss', color='orange')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('VAE Proxy to Label Cross Entropy')
    axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].set_xticks([])
    xlim = axes[0].get_xlim()
    axes[1].plot(range(nb_minibatches, nb_minibatches * epochs + 1, nb_minibatches),
                 valid_accuracies, label='Validation Accuracy', color='green')
    axes[1].set_xlabel('Batch Iterations')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('VAE Proxy to Label Accuracy')
    axes[1].legend()
    axes[1].set_xscale('log')
    axes[1].set_xlim(xlim)
    plt.tight_layout()

    plt.savefig(os.path.join(ckpt_dir, "groupifier_losses.png"))
    plt.close()
