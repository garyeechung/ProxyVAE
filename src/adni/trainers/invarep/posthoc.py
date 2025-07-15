import os

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import wandb

from src.adni.models import VariationalPredictor


WANDB_PROJECT = "InvaRep"
WANDB_ENTITY = "garyeechung-vanderbilt-university"
WANDB_GROUP = "ADNI_ResNet18"


def train_model(model: VariationalPredictor, train_loader,
                x_key, y_key, optimizer, loss_fn, device):
    model.train()

    total_losses = 0.0

    for batch in train_loader:
        x = batch[x_key].float().to(device)
        y = batch[y_key].float().to(device)

        optimizer.zero_grad()
        y_pred, _, _ = model(x)
        # suppose loss_fn is CrossEntropyLoss, y and y_pred are both float
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_losses += loss.item()
    avg_loss = total_losses / len(train_loader)
    return avg_loss


def evaluate_model(model: VariationalPredictor, valid_loader,
                   x_key, y_key, loss_fn, device):
    model.eval()

    total_losses = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            x = batch[x_key].float().to(device)
            y = batch[y_key].float().to(device)

            y_pred, _, _ = model(x)
            loss = loss_fn(y_pred, y)
            total_losses += loss.item()
    avg_loss = total_losses / len(valid_loader)
    return avg_loss


def train_posthoc_predictor(model: VariationalPredictor,
                            train_loader, valid_loader,
                            ckpt_dir: str, x_key: str, y_key: str,
                            beta1: float, beta2: float, device: str,
                            bootstrap: bool, epochs: int = 500,
                            lr: float = 5e-4,
                            if_existing_ckpt: str = "resume"):
    batch_size, _, h, w = next(iter(train_loader))[x_key].shape
    batch_per_epoch = len(train_loader)
    config = {
        "model_type": f"posthoc_{y_key}",
        "target_key": y_key,
        "beta1": beta1,
        "beta2": beta2,
        "lr": lr,
        "batch_size": batch_size,
        "input_shape": (h, w),
        "bootstrap": bootstrap,
        "batch_per_epoch": batch_per_epoch,
    }

    ckpt_dir = os.path.join(ckpt_dir, "invarep", f"beta1_{beta1:.1E}", f"beta2_{beta2:.1E}")

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    ckpt_path = os.path.join(ckpt_dir, f"posthoc_{y_key}.pth")
    ckpt_best_path = os.path.join(ckpt_dir, f"posthoc_{y_key}_best.pth")

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

    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, group=WANDB_GROUP,
               name=f"posthoc_{y_key}_beta1_{beta1:.1E}_beta2_{beta2:.1E}",
               config=config)

    model = model.to(device)

    for epoch in tqdm(range(ckpt_epoch + 1, ckpt_epoch + epochs + 1)):
        train_loss = train_model(model=model, train_loader=train_loader,
                                 x_key=x_key, y_key=y_key,
                                 optimizer=optimizer, loss_fn=loss_fn,
                                 device=device)

        valid_loss = evaluate_model(model=model, valid_loader=valid_loader,
                                    x_key=x_key, y_key=y_key,
                                    loss_fn=loss_fn, device=device)

        log_data = {
            "train/ce_loss": train_loss,
            "valid/ce_loss": valid_loss
        }

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_valid_loss": best_valid_loss,
                "epoch": epoch
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
