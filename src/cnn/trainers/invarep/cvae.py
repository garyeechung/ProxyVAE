import os

# import pandas as pd
import torch
from tqdm import tqdm
import wandb

from src.cnn.models import ConditionalVAE
from src.cnn.losses import VAE_Loss
from src.cnn.trainers.utils import vis_x_recon_comparison


WANDB_PROJECT = "InvaRep"
WANDB_ENTITY = "garyeechung-vanderbilt-university"


def train_model(model: ConditionalVAE, train_loader,
                x_key, y_key, optimizer, loss_fn, device):
    model.train()

    total_losses = 0.0
    kl_losses = 0.0
    recon_losses = 0.0
    for batch in train_loader:
        x = batch[x_key].float().to(device)
        y = batch[y_key].float().to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, y)
        total_loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, logvar)
        total_loss.backward()
        optimizer.step()

        total_losses += total_loss.item()
        kl_losses += kl_loss.item()
        recon_losses += recon_loss.item()

    avg_total_loss = total_losses / len(train_loader)
    avg_kl_loss = kl_losses / len(train_loader)
    avg_recon_loss = recon_losses / len(train_loader)
    return avg_total_loss, avg_recon_loss, avg_kl_loss


def evaluate_model(model: ConditionalVAE, val_loader,
                   x_key, y_key, loss_fn, device,
                   num_channels: int = 1,
                   return_comparison=True):
    model.eval()

    total_losses = 0.0
    kl_losses = 0.0
    recon_losses = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[x_key].float().to(device)
            y = batch[y_key].float().to(device)

            recon_x, mu, logvar = model(x, y)
            total_loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, logvar)

            total_losses += total_loss.item()
            kl_losses += kl_loss.item()
            recon_losses += recon_loss.item()

    avg_loss = total_losses / len(val_loader)
    avg_kl_loss = kl_losses / len(val_loader)
    avg_recon_loss = recon_losses / len(val_loader)

    if return_comparison:
        comparison = vis_x_recon_comparison(x.detach().cpu(), recon_x.detach().cpu(),
                                            n=4, num_channels=num_channels)
        return avg_loss, avg_recon_loss, avg_kl_loss, comparison
    else:
        return avg_loss, avg_recon_loss, avg_kl_loss


def train_cvae(model: ConditionalVAE, train_loader, valid_loader, ckpt_dir: str,
               x_key: str, y_key: str, beta1: float, device: str,
               dataset_name: str, backbone: str,
               bound_z_by: str, bootstrap: bool, epochs: int = 500,
               lr: float = 5e-4, if_existing_ckpt: str = "resume",
               num_channels: int = 1):
    """
    data_dir: Absolute path containing the ADNI data.
    ckpt_dir: Absolute path to save the checkpoints.
    x_key: Key for the input data in the batch.
    y_key: Key for the target data in the batch.
    beta1: Weight for the KL divergence loss.
    """

    batch_size, _, h, w = next(iter(train_loader))[x_key].shape
    batch_per_epoch = len(train_loader)
    config = {
        "model_type": "cvae",
        "target_key": y_key,
        "beta1": beta1,
        "lr": lr,
        "batch_size": batch_size,
        "input_shape": (h, w),
        "bootstrap": bootstrap,
        "batch_per_epoch": batch_per_epoch,
        "bound_z_by": bound_z_by,
    }

    ckpt_dir = os.path.join(ckpt_dir, "invarep", f"beta1_{beta1:.1E}")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = VAE_Loss(beta1)

    ckpt_path = os.path.join(ckpt_dir, "cvae.pth")
    ckpt_best_path = os.path.join(ckpt_dir, "cvae_best.pth")
    if os.path.exists(ckpt_path) and if_existing_ckpt == "pass":
        checkpoint = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    elif os.path.exists(ckpt_path) and if_existing_ckpt == "resume":
        checkpoint = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_valid_loss = checkpoint["best_valid_loss"]

        ckpt_epoch = checkpoint["epoch"]
        if ckpt_epoch >= epochs:
            return model
        else:
            epochs -= ckpt_epoch
    elif os.path.exists(ckpt_path) and if_existing_ckpt == "replace":
        os.remove(ckpt_path)
        best_valid_loss = float("inf")
        ckpt_epoch = 0
    else:
        best_valid_loss = float("inf")
        ckpt_epoch = 0

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               group=f"{dataset_name}_{backbone}", name=f"cvae_beta1_{beta1:.1E}",
               config=config)

    model = model.to(device)

    for epoch in tqdm(range(ckpt_epoch + 1, ckpt_epoch + epochs + 1)):
        train_total_loss, train_recon_loss, train_kl_loss = train_model(
            model=model, train_loader=train_loader,
            x_key=x_key, y_key=y_key, optimizer=optimizer,
            loss_fn=loss_fn, device=device)

        valid_total_loss, valid_recon_loss, valid_kl_loss, comparison = evaluate_model(
            model=model, val_loader=valid_loader,
            x_key=x_key, y_key=y_key, num_channels=num_channels,
            loss_fn=loss_fn, device=device, return_comparison=True)

        log_data = {
            "train/total_loss": train_total_loss,
            "train/kl_loss": train_kl_loss,
            "train/recon_loss": train_recon_loss,
            "valid/total_loss": valid_total_loss,
            "valid/kl_loss": valid_kl_loss,
            "valid/recon_loss": valid_recon_loss,
        }

        if valid_total_loss < best_valid_loss:
            best_valid_loss = valid_total_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss
            }, ckpt_best_path)
            log_data["valid/comparison"] = wandb.Image(comparison, caption=f"Epoch {epoch}")

        wandb.log(log_data)

    latest_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_valid_loss": best_valid_loss
    }
    torch.save(latest_checkpoint, ckpt_path)
    wandb.finish()
    return model
