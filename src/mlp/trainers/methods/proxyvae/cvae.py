import os

import numpy as np
import torch
from tqdm import tqdm
import wandb

from src.mlp.models import ConditionalVAE
from src.mlp.losses import VAE_Loss
from src.mlp.trainers.utils import plot_tsne


WANDB_PROJECT = "ProxyVAE"
WANDB_ENTITY = "garyeechung-vanderbilt-university"


def train_model(model: ConditionalVAE, train_loader,
                x_key, y_key, optimizer, loss_fn, device, y_grain="coarse"):
    model.train()

    total_losses = 0.0
    kl_losses = 0.0
    recon_losses = 0.0
    for batch in train_loader:
        x = batch[0].float().to(device)
        if y_grain == "coarse":
            y = batch[1].float().to(device)
        elif y_grain == "fine":
            y = batch[2].float().to(device)
        else:
            raise ValueError(f"y_grain {y_grain} not recognized.")

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x, y)
        total_loss, recon_loss, kl_loss = loss_fn(x, recon_x, mu, logvar)
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
                   tsne_config: dict, y_grain="coarse",
                   comparison_fn=None):
    model.eval()

    total_losses = 0.0
    kl_losses = 0.0
    recon_losses = 0.0
    z_all = []
    yc_all = []
    yf_all = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].float().to(device)
            x = batch[0].float().to(device)
            if y_grain == "coarse":
                y = batch[1].float().to(device)
            elif y_grain == "fine":
                y = batch[2].float().to(device)
            else:
                raise ValueError(f"y_grain {y_grain} not recognized.")
            yc_all.append(batch[1].cpu().numpy().argmax(axis=1))
            yf_all.append(batch[2].cpu().numpy().argmax(axis=1))

            recon_x, mu, logvar = model(x, y)
            total_loss, recon_loss, kl_loss = loss_fn(recon_x, x, mu, logvar)
            total_losses += total_loss.item()
            kl_losses += kl_loss.item()
            recon_losses += recon_loss.item()

            z_all.append(mu.detach().cpu().numpy())

    avg_total_loss = total_losses / len(val_loader)
    avg_kl_loss = kl_losses / len(val_loader)
    avg_recon_loss = recon_losses / len(val_loader)

    z_all = np.concatenate(z_all, axis=0)
    yc_all = np.concatenate(yc_all, axis=0)
    yf_all = np.concatenate(yf_all, axis=0)

    tsne_img = plot_tsne(z_all, yc_all, yf_all, **tsne_config)

    comparison = None
    if comparison_fn is not None:
        comparison = comparison_fn(x.cpu()[0], recon_x.cpu()[0])

    return avg_total_loss, avg_recon_loss, avg_kl_loss, comparison, tsne_img


def train_cvae(model: ConditionalVAE, train_loader, valid_loader, ckpt_dir: str,
               x_key: str, y_key: str, beta1: float, device: str,
               dataset_name: str, bound_z_by: str,
               tsne_config: dict, epochs: int = 500,
               lr: float = 5e-3, if_existing_ckpt: str = "resume", comparison_fn=None,
               y_grain="coarse"):

    batch_size = next(iter(train_loader))[0].shape[0]
    batch_per_epoch = len(train_loader)
    config = {
        "model_type": "CVAE",
        "target_key": y_key,
        "beta1": beta1,
        "lr": lr,
        "batch_size": batch_size,
        "batch_per_epoch": batch_per_epoch,
        "bound_z_by": bound_z_by,
    }

    ckpt_dir = os.path.join(ckpt_dir, "proxyvae", f"beta1_{beta1:.1E}")
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

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=f"{dataset_name}",
               name=f"cvae_beta1_{beta1:.1E}", config=config)

    model = model.to(device)

    for epoch in tqdm(range(ckpt_epoch + 1, ckpt_epoch + epochs + 1)):
        train_total_loss, train_recon_loss, train_kl_loss = train_model(
            model=model, train_loader=train_loader, x_key=x_key, y_key=y_key,
            optimizer=optimizer, loss_fn=loss_fn, device=device, y_grain=y_grain)

        valid_total_loss, valid_recon_loss, valid_kl_loss, comparison, tsne_img = evaluate_model(
            model=model, val_loader=valid_loader, x_key=x_key, y_key=y_key,
            loss_fn=loss_fn, device=device, comparison_fn=comparison_fn, tsne_config=tsne_config, y_grain=y_grain)

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
            if comparison is not None:
                log_data["valid/comparison"] = wandb.Image(comparison, caption=f"Epoch {epoch}")
            log_data["valid/tsne"] = wandb.Image(tsne_img, caption=f"Epoch {epoch}")

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
