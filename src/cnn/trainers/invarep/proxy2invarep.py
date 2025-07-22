import os

import torch
from torch.nn import MSELoss
from tqdm import tqdm
import wandb

from src.cnn.models import ProxyRep2InvaRep


WANDB_PROJECT = "InvaRep"
WANDB_ENTITY = "garyeechung-vanderbilt-university"


def train_model(model: ProxyRep2InvaRep, train_loader,
                x_key, optimizer, loss_fn, device) -> None:
    model.train()

    total_losses = 0.0
    for batch in train_loader:
        x = batch[x_key].float().to(device)

        optimizer.zero_grad()
        z1, z2, z1_pred = model(x)
        loss = loss_fn(z1_pred, z1)
        loss.backward()
        optimizer.step()

        total_losses += loss.item()
    avg_loss = total_losses / len(train_loader)

    return avg_loss


def evaluate_model(model: ProxyRep2InvaRep, valid_loader,
                   x_key, loss_fn, device) -> float:
    model.eval()
    total_losses = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            x = batch[x_key].float().to(device)
            z1, z2, z1_pred = model(x)
            loss = loss_fn(z1_pred, z1)
            total_losses += loss.item()
    avg_loss = total_losses / len(valid_loader)
    return avg_loss


def train_proxy2invarep(model: ProxyRep2InvaRep, train_loader, valid_loader,
                        ckpt_dir: str, x_key: str, device: str,
                        beta1: float, beta2: float, bootstrap: bool,
                        bound_z_by: str, dataset_name: str, backbone: str,
                        epochs: int = 500, lr: float = 5e-4,
                        if_existing_ckpt: str = "resume"):
    batch_size, _, h, w = next(iter(train_loader))[x_key].shape
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
        "bound_z_by": bound_z_by,
    }

    ckpt_dir = os.path.join(ckpt_dir, "invarep", f"beta1_{beta1:.1E}", f"beta2_{beta2:.1E}")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    ckpt_path = os.path.join(ckpt_dir, "proxy2invarep.pth")
    ckpt_best_path = os.path.join(ckpt_dir, "proxy2invarep_best.pth")
    if os.path.exists(ckpt_path) and if_existing_ckpt == "pass":
        checkpoint = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        ckpt_epoch = checkpoint["epoch"]
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
               group=f"{dataset_name}_{backbone}",
               name=f"proxy2invarep_beta1_{beta1:.1E}_beta2_{beta2:.1E}",
               config=config)

    model = model.to(device)

    for epoch in tqdm(range(ckpt_epoch + 1, ckpt_epoch + epochs + 1)):
        train_loss = train_model(model=model, train_loader=train_loader,
                                 x_key=x_key, optimizer=optimizer,
                                 loss_fn=loss_fn, device=device)
        valid_loss = evaluate_model(model=model, valid_loader=valid_loader,
                                    x_key=x_key, loss_fn=loss_fn, device=device)

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
