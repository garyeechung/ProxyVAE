import os

import torch
from torch.nn import MSELoss
from tqdm import tqdm
import wandb

from src.adni.models import ProxyRep2InvaRep


WANDB_PROJECT = "InvaRep"
WANDB_ENTITY = "garyeechung-vanderbilt-university"
WANDB_GROUP = "ADNI_by_manufacturer"


def train_model(model: ProxyRep2InvaRep, train_loader,
                optimizer, loss_fn, device) -> None:
    model.train()

    total_losses = 0.0
    for batch in train_loader:
        x = batch["image"].float().to(device)

        optimizer.zero_grad()
        z1, z2, z1_pred = model(x)
        loss = loss_fn(z1_pred, z1)
        loss.backward()
        optimizer.step()

        total_losses += loss.item()
    avg_loss = total_losses / len(train_loader)

    return avg_loss


def evaluate_model(model: ProxyRep2InvaRep, valid_loader, loss_fn, device) -> float:
    model.eval()
    total_losses = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            x = batch["image"].float().to(device)
            z1, z2, z1_pred = model(x)
            loss = loss_fn(z1_pred, z1)
            total_losses += loss.item()
    avg_loss = total_losses / len(valid_loader)
    return avg_loss


def train_proxy2invarep(model: ProxyRep2InvaRep, train_loader, valid_loader,
                        ckpt_dir: str, device: str,
                        beta1: float, beta2: float, bootstrap: bool,  # this three for config only, must match the model
                        epochs: int = 500, lr: float = 5e-4,
                        if_existing_ckpt: str = "resume"):
    batch_size, _, h, w = next(iter(train_loader))["image"].shape
    batch_per_epoch = len(train_loader)
    config = {
        "model_type": "proxy2invarep",
        "beta1": beta1,
        "beta2": beta2,
        "lr": lr,
        "batch_size": batch_size,
        "input_shape": (h, w),
        "bootstrap": bootstrap,
        "batch_per_epoch": batch_per_epoch,
    }

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               group=WANDB_GROUP, name=f"proxy2invarep_beta1_{beta1:.1E}_beta2_{beta2:.1E}",
               config=config)

    ckpt_dir = os.path.join(ckpt_dir, "invarep", f"beta1_{beta1}", f"beta2_{beta2}")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    ckpt_path = os.path.join(ckpt_dir, "proxy2invarep.pth")
    ckpt_best_path = os.path.join(ckpt_dir, "proxy2invarep_best.pth")
    if os.path.exists(ckpt_path) and if_existing_ckpt == "pass":
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model
    elif os.path.exists(ckpt_path) and if_existing_ckpt == "resume":
        checkpoint = torch.load(ckpt_path)
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
    else:
        best_valid_loss = float("inf")

    model = model.to(device)

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train_model(model=model, train_loader=train_loader,
                                 optimizer=optimizer, loss_fn=loss_fn,
                                 device=device)
        valid_loss = evaluate_model(model=model, valid_loader=valid_loader,
                                    loss_fn=loss_fn, device=device)

        log_data = {
            "train/mse_loss": train_loss,
            "valid/mse_loss": valid_loss,
        }

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss
            }, ckpt_best_path)
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
