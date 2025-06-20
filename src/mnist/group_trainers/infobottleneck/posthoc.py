import os

import matplotlib.pyplot as plt
import torch
from torch.nn import Module
import torch.optim as optim
from tqdm import tqdm

from src.mnist.datasets import get_mnist_dataloaders


def train_posthoc_model(model, train_loader, optimizer, loss_fn, device,
                        return_each_batch=True):
    model.train()
    batch_losses = []

    for x, y in train_loader:
        x = x.to(device)
        y = y.float().to(device)
        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())

    if return_each_batch:
        return batch_losses
    else:
        return [sum(batch_losses) / len(train_loader)]


def evaluate_posthoc_model(model, valid_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.float().to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

            y_pred = torch.argmax(y_pred, dim=-1)
            y = torch.argmax(y, dim=-1)
            correct += (y_pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total if total > 0 else 0.0
    return total_loss / len(valid_loader), accuracy


def train_infobottleneck_posthoc(posthoc_model: Module,
                                 data_dir: str, ckpt_dir: str,
                                 beta: float, device: str, batch_size=500,
                                 lr=5e-4, epochs=200, return_each_batch=True,
                                 replace_existing_ckpt=False):
    ckpt_dir = os.path.join(ckpt_dir, "information_bottleneck", f"beta_{beta}")
    ckpt_path = os.path.join(ckpt_dir, "posthoc.pth")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    elif not replace_existing_ckpt and os.path.exists(ckpt_path):
        print(f"Checkpoint already exists at {ckpt_path}. Skipping training.")
        return

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    dataloaders = get_mnist_dataloaders(data_dir, one_hot=True, batch_size=batch_size)
    train_loader, valid_loader, test_loader = dataloaders

    posthoc_model.to(device)
    optimizer = optim.Adam(posthoc_model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_loss, _ = evaluate_posthoc_model(model=posthoc_model,
                                              valid_loader=valid_loader,
                                              loss_fn=loss_fn, device=device)

    train_losses = []
    valid_losses = []
    valid_accuracies = []

    for epoch in tqdm(range(epochs)):
        train_loss = train_posthoc_model(model=posthoc_model, train_loader=train_loader,
                                         optimizer=optimizer, loss_fn=loss_fn, device=device,
                                         return_each_batch=return_each_batch)
        train_losses = train_losses + train_loss

        valid_loss, valid_acc = evaluate_posthoc_model(model=posthoc_model,
                                                       valid_loader=valid_loader,
                                                       loss_fn=loss_fn, device=device)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(posthoc_model.state_dict(), ckpt_path)

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

    plt.savefig(os.path.join(ckpt_dir, "posthoc_losses.png"))
    plt.close()
