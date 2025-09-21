import os

import numpy as np
import torch
from tqdm import tqdm
import wandb

from src.mlp.models import VariationalPredictor
from src.mlp.losses import VIB_Loss
from src.mlp.trainers.utils import plot_tsne

WANDB_PROJECT = "ProxyVAE"
WANDB_ENTITY = "garyeechung-vanderbilt-university"


def train_model(model: VariationalPredictor, train_loader,
                x_key, y_key, optimizer, is_posthoc: bool,
                loss_fn: VIB_Loss, device, y_grain="coarse"):
    model.train()

    total_losses = 0.0
    kl_losses = 0.0
    ce_losses = 0.0

    y_true_all = []
    y_pred_all = []
    for batch in train_loader:
        x = batch[0].float().to(device)
        if y_grain == "coarse":
            y = batch[1].argmax(dim=-1).long().to(device)
        elif y_grain == "fine":
            y = batch[2].argmax(dim=-1).long().to(device)
        else:
            raise ValueError(f"y_grain {y_grain} not recognized.")
        y_true_all.append(y.cpu().numpy())

        optimizer.zero_grad()
        y_pred, mu, logvar = model(x)
        y_pred_all.append(y_pred.detach().cpu().numpy().argmax(axis=-1))

        total_loss, ce_loss, kl_loss = loss_fn(y, y_pred, mu, logvar)
        if is_posthoc:
            ce_loss.backward()
        else:
            total_loss.backward()
        optimizer.step()
        total_losses += total_loss.item()
        kl_losses += kl_loss.item()
        ce_losses += ce_loss.item()
    avg_total_loss = total_losses / len(train_loader)
    avg_kl_loss = kl_losses / len(train_loader)
    avg_ce_loss = ce_losses / len(train_loader)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    accuracy = np.mean(y_true_all == y_pred_all)
    return avg_total_loss, avg_ce_loss, avg_kl_loss, accuracy


def evaluate_model(model: VariationalPredictor, val_loader,
                   x_key, y_key, loss_fn: VIB_Loss, device,
                   tsne_config: dict, y_grain="coarse"):
    model.eval()

    total_losses = 0.0
    ce_losses = 0.0
    kl_losses = 0.0
    z_all = []
    yc_all = []
    yf_all = []
    y_true_all = []
    y_pred_all = []
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].float().to(device)
            if y_grain == "coarse":
                y = batch[1].argmax(dim=-1).long().to(device)
            elif y_grain == "fine":
                y = batch[2].argmax(dim=-1).long().to(device)
            else:
                raise ValueError(f"y_grain {y_grain} not recognized.")
            y_true_all.append(y.cpu().numpy())

            yc_all.append(batch[1].cpu().numpy().argmax(axis=1))
            yf_all.append(batch[2].cpu().numpy().argmax(axis=1))

            y_pred, mu, logvar = model(x)
            y_pred_all.append(y_pred.detach().cpu().numpy().argmax(axis=-1))

            total_loss, ce_loss, kl_loss = loss_fn(y, y_pred, mu, logvar)
            total_losses += total_loss.item()
            ce_losses += ce_loss.item()
            kl_losses += kl_loss.item()

            z_all.append(mu.cpu().numpy())
    avg_total_loss = total_losses / len(val_loader)
    avg_kl_loss = kl_losses / len(val_loader)
    avg_ce_loss = ce_losses / len(val_loader)

    z_all = np.concatenate(z_all, axis=0)
    yc_all = np.concatenate(yc_all, axis=0)
    yf_all = np.concatenate(yf_all, axis=0)

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    accuracy = np.mean(y_true_all == y_pred_all)

    tsne_img = plot_tsne(z_all, yc_all, yf_all, **tsne_config)

    return avg_total_loss, avg_ce_loss, avg_kl_loss, accuracy, tsne_img


def train_vib(model: VariationalPredictor, train_loader, valid_loader,
              ckpt_dir: str, x_key: str, y_key: str, beta: float, device: str,
              dataset_name: str, bound_z_by: str, is_posthoc: bool,
              tsne_config: dict, epochs: int = 500, lr: float = 5e-3,
              if_existing_ckpt: str = "resume", y_grain="coarse"):
    batch_size = next(iter(train_loader))[0].shape[0]
    batch_per_epoch = len(train_loader)
    config = {
        "model_type": "VIB",
        "target": y_key,
        "beta": beta,
        "lr": lr,
        "batch_size": batch_size,
        "batch_per_epoch": batch_per_epoch,
        "bound_z_by": bound_z_by,
    }

    ckpt_dir = os.path.join(ckpt_dir, "vib", f"beta_{beta:.1E}")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = VIB_Loss(beta=beta)

    ckpt_path = os.path.join(ckpt_dir, f"vib_{y_grain}.pth")
    ckpt_best_path = os.path.join(ckpt_dir, f"vib_{y_grain}_best.pth")
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
               name=f"vib_{y_grain}_beta_{beta:.1E}", config=config)

    model = model.to(device)

    for epoch in tqdm(range(ckpt_epoch + 1, ckpt_epoch + epochs + 1)):
        train_total_loss, train_ce_loss, train_kl_loss, train_accuracy = train_model(
            model=model, train_loader=train_loader, x_key=x_key, y_key=y_key,
            optimizer=optimizer, loss_fn=loss_fn, device=device, y_grain=y_grain, is_posthoc=is_posthoc)

        valid_total_loss, valid_ce_loss, valid_kl_loss, valid_accuracy, tsne_img = evaluate_model(
            model=model, val_loader=valid_loader, x_key=x_key, y_key=y_key,
            loss_fn=loss_fn, device=device, tsne_config=tsne_config, y_grain=y_grain)

        log_data = {
            "train/total_loss": train_total_loss,
            "train/ce_loss": train_ce_loss,
            "train/kl_loss": train_kl_loss,
            "train/accuracy": train_accuracy,
            "valid/total_loss": valid_total_loss,
            "valid/ce_loss": valid_ce_loss,
            "valid/kl_loss": valid_kl_loss,
            "valid/accuracy": valid_accuracy,
        }

        if valid_total_loss < best_valid_loss:
            best_valid_loss = valid_total_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss
            }, ckpt_best_path)

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
